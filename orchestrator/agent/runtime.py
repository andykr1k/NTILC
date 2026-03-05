"""Full NTILC orchestrator runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence

from models.software_layer import DispatchResult, ToolDispatcher
from orchestrator.blocks import build_dispatch_block, build_plan_block, build_response_block
from orchestrator.planning import (
    PlanAction,
    actions_to_instruction_list,
    enforce_atomic_actions,
    parse_plan_block,
    salvage_plan_actions,
)
from orchestrator.results import OrchestratorRunResult, OrchestratorStepResult, RetrievalCandidate

if TYPE_CHECKING:  # pragma: no cover
    from orchestrator.generation.model import QwenOrchestratorModel
    from orchestrator.retrieval.system import ClusterBasedToolSystem


class NTILCOrchestratorAgent:
    """
    Full NTILC inference controller:
    plan -> retrieval -> tool mapping -> command generation -> dispatch.
    """

    def __init__(
        self,
        retrieval_system: Any,
        dispatcher: ToolDispatcher,
        qwen_model: Optional[Any] = None,
    ):
        self.retrieval_system = retrieval_system
        self.dispatcher = dispatcher
        self.qwen_model = qwen_model

    @classmethod
    def from_pretrained(
        cls,
        intent_embedder_path: Optional[str],
        query_encoder_path: str,
        qwen_model_name_or_path: Optional[str] = None,
        lora_adapter_path: Optional[str] = None,
        lora_mode: str = "full",
        auto_register_shell_tools: bool = True,
        tool_timeout_seconds: int = 20,
        tool_cwd: Optional[str] = None,
        fail_on_nonzero_exit: bool = True,
        device: Optional[str] = None,
    ) -> "NTILCOrchestratorAgent":
        from orchestrator.retrieval.system import ClusterBasedToolSystem

        retrieval_system = ClusterBasedToolSystem.from_pretrained(
            intent_embedder_path=intent_embedder_path,
            query_encoder_path=query_encoder_path,
            device=device,
        )

        mapper = retrieval_system.mapper
        if auto_register_shell_tools:
            mapper.register_shell_tools_for_all_clusters(
                timeout_seconds=tool_timeout_seconds,
                cwd=tool_cwd,
            )
        dispatcher = ToolDispatcher(mapper=mapper, fail_on_nonzero_exit=fail_on_nonzero_exit)

        qwen_model: Optional[Any] = None
        if qwen_model_name_or_path:
            from orchestrator.generation.model import QwenOrchestratorModel

            qwen_model = QwenOrchestratorModel.from_pretrained(
                qwen_model_name_or_path=qwen_model_name_or_path,
                lora_adapter_path=lora_adapter_path,
                mode=lora_mode,
            )

        return cls(
            retrieval_system=retrieval_system,
            dispatcher=dispatcher,
            qwen_model=qwen_model,
        )

    def _build_actions(
        self,
        request: str,
        max_plan_actions: int,
        plan_max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, List[PlanAction]]:
        parsed_actions: List[PlanAction] = []
        raw_plan_block = ""

        if self.qwen_model is not None and hasattr(self.qwen_model, "generate_plan_actions"):
            payload = self.qwen_model.generate_plan_actions(
                request=request,
                max_actions=max_plan_actions,
                max_new_tokens=plan_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            raw_plan_block = str(payload.get("plan_block", "")).strip()
            action_texts = [
                str(action).strip()
                for action in payload.get("actions", [])
                if str(action).strip()
            ]
            if action_texts:
                parsed_actions = [
                    PlanAction(id=idx, instruction=instruction)
                    for idx, instruction in enumerate(action_texts, start=1)
                ]
        elif self.qwen_model is not None:
            raw_plan_block = self.qwen_model.generate_plan(
                request=request,
                max_actions=max_plan_actions,
                max_new_tokens=plan_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            raw_plan_block = build_plan_block([request])

        if not parsed_actions:
            try:
                parsed_actions = parse_plan_block(raw_plan_block)
            except ValueError:
                parsed_actions = salvage_plan_actions(raw_plan_block, fallback_request=request)

        atomic_actions = enforce_atomic_actions(parsed_actions)
        if not atomic_actions:
            atomic_actions = salvage_plan_actions(request, fallback_request=request)
        if not atomic_actions:
            fallback = str(request).strip()
            atomic_actions = [PlanAction(id=1, instruction=fallback)] if fallback else []

        canonical_plan = build_plan_block(actions_to_instruction_list(atomic_actions))
        return canonical_plan, atomic_actions

    @staticmethod
    def _compact_prior_step_summaries(steps: Sequence[OrchestratorStepResult], tail_k: int = 6) -> List[str]:
        summaries: List[str] = []
        for step in steps[-max(0, int(tail_k)) :]:
            action_id = step.action_id if step.action_id is not None else "?"
            status = "ok" if step.dispatch_result.ok else "fail"
            summaries.append(f"action#{action_id} {step.action_text} -> {step.command} [{status}]")
        return summaries

    def _append_failed_action_step(
        self,
        run_result: OrchestratorRunResult,
        action: PlanAction,
        error_text: str,
    ) -> None:
        empty_dispatch = DispatchResult(
            ok=False,
            tool="",
            arguments={},
            result=None,
            errors=[error_text],
            executed=False,
        )
        step = OrchestratorStepResult(
            candidate=RetrievalCandidate(cluster_id=-1, score=0.0, tool_name=""),
            generated_text="",
            command="",
            dispatch_arguments={},
            dispatch_result=empty_dispatch,
            dispatch_block=build_dispatch_block("", {}),
            response_block=build_response_block("", empty_dispatch, retry=False),
            action_id=action.id,
            action_text=action.instruction,
        )
        run_result.steps.append(step)

    def run(
        self,
        request: str,
        execute_tools: bool = False,
        top_k_candidates: int = 3,
        max_retries: int = 2,
        similarity_threshold: float = -1.0,
        granted_permissions: Optional[Iterable[str]] = None,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_plan_actions: int = 8,
        plan_max_new_tokens: int = 256,
    ) -> OrchestratorRunResult:
        plan_block, atomic_actions = self._build_actions(
            request=request,
            max_plan_actions=max_plan_actions,
            plan_max_new_tokens=plan_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        run_result = OrchestratorRunResult(
            request=request,
            plan_block=plan_block,
            candidates=[],
            steps=[],
            atomic_actions=[action.to_dict() for action in atomic_actions],
            action_failures=[],
        )

        for action in atomic_actions:
            candidates = self.retrieval_system.retrieve_candidates(
                query=action.instruction,
                top_k=top_k_candidates,
                threshold=similarity_threshold,
            )
            run_result.candidates.extend(candidates)

            if not candidates:
                reason = f"No candidates retrieved for action `{action.instruction}`."
                self._append_failed_action_step(run_result, action=action, error_text=reason)
                run_result.action_failures.append(
                    {
                        "action_id": action.id,
                        "action_text": action.instruction,
                        "reason": reason,
                    }
                )
                break

            max_attempts = min(len(candidates), max(1, int(max_retries) + 1))
            action_succeeded = False

            for attempt_idx in range(max_attempts):
                candidate = candidates[attempt_idx]

                if self.qwen_model is not None:
                    if hasattr(self.qwen_model, "generate_dispatch_arguments"):
                        generated = self.qwen_model.generate_dispatch_arguments(
                            query=request,
                            tool=candidate.tool_name,
                            current_action=action.instruction,
                            prior_step_summaries=self._compact_prior_step_summaries(run_result.steps),
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        generated_text = str(generated.get("generated_text", "")).strip()
                        dispatch_arguments = dict(generated.get("arguments", {}) or {})
                        command = str(dispatch_arguments.get("command", "")).strip()
                    else:
                        generated = self.qwen_model.generate_command(
                            query=request,
                            tool=candidate.tool_name,
                            current_action=action.instruction,
                            prior_step_summaries=self._compact_prior_step_summaries(run_result.steps),
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        generated_text = generated["generated_text"]
                        command = generated["command"]
                        dispatch_arguments = {
                            "command": command,
                            "query": request,
                        }
                else:
                    generated_text = candidate.tool_name
                    command = candidate.tool_name
                    dispatch_arguments = {
                        "command": command,
                        "query": request,
                    }

                if not command:
                    command = candidate.tool_name
                    dispatch_arguments["command"] = command
                dispatch_arguments.setdefault("query", request)
                dispatch_result = self.dispatcher.dispatch_cluster(
                    cluster_id=candidate.cluster_id,
                    arguments=dispatch_arguments,
                    execute=execute_tools,
                    granted_permissions=granted_permissions,
                )

                has_next_attempt = (attempt_idx + 1) < max_attempts
                step = OrchestratorStepResult(
                    candidate=candidate,
                    generated_text=generated_text,
                    command=command,
                    dispatch_arguments=dispatch_arguments,
                    dispatch_result=dispatch_result,
                    dispatch_block=build_dispatch_block(candidate.tool_name, dispatch_arguments),
                    response_block=build_response_block(
                        candidate.tool_name,
                        dispatch_result,
                        retry=has_next_attempt and not dispatch_result.ok,
                    ),
                    action_id=action.id,
                    action_text=action.instruction,
                )
                run_result.steps.append(step)

                if dispatch_result.ok:
                    action_succeeded = True
                    break

            if not action_succeeded:
                last_step = run_result.steps[-1] if run_result.steps else None
                if last_step and last_step.dispatch_result.errors:
                    reason = "; ".join(last_step.dispatch_result.errors)
                else:
                    reason = f"Action `{action.instruction}` failed after retries."
                run_result.action_failures.append(
                    {
                        "action_id": action.id,
                        "action_text": action.instruction,
                        "reason": reason,
                    }
                )
                break

        return run_result

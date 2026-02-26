"""
End-to-end orchestrator agent for NTILC.

Pipeline:
user request -> Qwen atomic planner -> cluster retrieval -> cluster->tool mapping
-> Qwen argument fill -> tool call string -> optional direct execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference import ClusterBasedToolSystem, format_tool_call
from models.argument_inference import ArgumentValueGenerator
from models.tool_schemas import TOOL_SCHEMAS


ATOMIC_PLAN_SYSTEM_PROMPT = """
You are an orchestration planner.
Break the user request into atomic, executable actions.
Each action must represent one intent that should map to one tool call.
Return only valid JSON with this schema:
{
  "actions": [
    {"id": 1, "instruction": "atomic action text", "depends_on": []}
  ]
}
Rules:
- No prose outside JSON.
- Keep actions minimal and concrete.
- Use 1..N integer ids.
- Use depends_on ids only when an action needs a previous action's output.
""".strip()


ARG_FILL_SYSTEM_PROMPT = """
You fill tool arguments from an atomic action.
Return only valid JSON object of arguments with keys from the tool schema.
If a value is unknown, set it to null.
Never include explanation text.
""".strip()


@dataclass
class OrchestratorStepResult:
    step_id: int
    action: str
    cluster_id: int
    confidence: float
    tool_name: Optional[str]
    arguments: Dict[str, Any]
    tool_call: Optional[str]
    status: str
    validation_errors: List[str] = field(default_factory=list)
    execution: Optional[Dict[str, Any]] = None


@dataclass
class OrchestratorRunResult:
    request: str
    atomic_actions: List[Dict[str, Any]]
    steps: List[OrchestratorStepResult]
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request,
            "atomic_actions": self.atomic_actions,
            "steps": [s.__dict__ for s in self.steps],
            "duration_ms": self.duration_ms,
        }


class QwenOrchestratorModel:
    """
    Thin Qwen wrapper for:
    1) atomic action planning
    2) argument completion for resolved tools.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
        torch_dtype: str = "float16",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        if device is None:
            device = "cuda:7" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            lines = []
            for message in messages:
                role = message.get("role", "user").upper()
                content = message.get("content", "")
                lines.append(f"{role}: {content}")
            lines.append("ASSISTANT:")
            prompt = "\n".join(lines)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def _extract_json(text: str) -> Optional[Any]:
        text = text.strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract the first balanced JSON object/array if model adds extra text.
        for open_char, close_char in [("{", "}"), ("[", "]")]:
            starts = [idx for idx, char in enumerate(text) if char == open_char]
            for start in starts:
                depth = 0
                for idx in range(start, len(text)):
                    char = text[idx]
                    if char == open_char:
                        depth += 1
                    elif char == close_char:
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:idx + 1]
                            try:
                                return json.loads(candidate)
                            except json.JSONDecodeError:
                                break
        return None

    def plan_atomic_actions(
        self,
        request: str,
        max_actions: int = 8,
    ) -> List[Dict[str, Any]]:
        user_prompt = (
            f"User request:\n{request}\n\n"
            f"Return at most {max_actions} actions."
        )
        raw = self._generate([
            {"role": "system", "content": ATOMIC_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        parsed = self._extract_json(raw)

        actions: List[Dict[str, Any]] = []
        if isinstance(parsed, dict):
            raw_actions = parsed.get("actions", [])
        elif isinstance(parsed, list):
            raw_actions = parsed
        else:
            raw_actions = []

        for idx, action in enumerate(raw_actions, start=1):
            if isinstance(action, dict):
                instruction = str(action.get("instruction", "")).strip()
                depends_on = action.get("depends_on", [])
            else:
                instruction = str(action).strip()
                depends_on = []

            if not instruction:
                continue

            if not isinstance(depends_on, list):
                depends_on = []
            depends_on = [int(x) for x in depends_on if str(x).isdigit()]

            actions.append({
                "id": idx,
                "instruction": instruction,
                "depends_on": depends_on,
            })
            if len(actions) >= max_actions:
                break

        if not actions:
            # Hard fallback: one atomic action as the original request.
            actions = [{"id": 1, "instruction": request.strip(), "depends_on": []}]

        return actions

    def infer_arguments(
        self,
        action: str,
        tool_name: str,
        tool_schema: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        schema_json = json.dumps(tool_schema.get("parameters", {}), ensure_ascii=True)
        context_json = json.dumps(context, ensure_ascii=True)
        user_prompt = (
            f"Atomic action:\n{action}\n\n"
            f"Tool: {tool_name}\n"
            f"Tool parameters schema:\n{schema_json}\n\n"
            f"Context:\n{context_json}\n\n"
            "Return JSON object arguments only."
        )
        raw = self._generate([
            {"role": "system", "content": ARG_FILL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        parsed = self._extract_json(raw)

        if not isinstance(parsed, dict):
            return {}

        # Some models wrap in {"arguments": {...}}
        if "arguments" in parsed and isinstance(parsed["arguments"], dict):
            parsed = parsed["arguments"]

        allowed = set(tool_schema.get("parameters", {}).keys())
        return {k: v for k, v in parsed.items() if k in allowed}


class NTILCOrchestratorAgent:
    """
    Full orchestration agent for testing:
    intent embedder + cluster retrieval + software layer.
    """

    def __init__(
        self,
        cluster_system: ClusterBasedToolSystem,
        planner: QwenOrchestratorModel,
        log_file: str = "logs/orchestrator_agent.log",
        log_level: int = logging.INFO,
    ):
        self.cluster_system = cluster_system
        self.planner = planner
        self.arg_fallback = ArgumentValueGenerator()
        self.logger = self._build_logger(
            name="orchestrator.agent",
            log_file=log_file,
            level=log_level,
        )

    @classmethod
    def from_pretrained(
        cls,
        intent_embedder_path: str,
        query_encoder_path: str,
        qwen_model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        mapper_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: str = "float16",
        log_file: str = "logs/orchestrator_agent.log",
    ) -> "NTILCOrchestratorAgent":
        cluster_system = ClusterBasedToolSystem.from_pretrained(
            intent_embedder_path=intent_embedder_path,
            query_encoder_path=query_encoder_path,
            mapper_path=mapper_path,
            device=device,
        )
        planner = QwenOrchestratorModel(
            model_name_or_path=qwen_model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        return cls(
            cluster_system=cluster_system,
            planner=planner,
            log_file=log_file,
        )

    @staticmethod
    def _build_logger(name: str, log_file: str, level: int) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        if logger.handlers:
            return logger

        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def register_tool_callable(self, tool_name: str, fn) -> None:
        self.cluster_system.mapper.register_tool_callable(tool_name, fn)

    def register_cluster_callable(self, cluster_id: int, fn) -> None:
        self.cluster_system.mapper.register_cluster_callable(cluster_id, fn)

    @staticmethod
    def _coerce_value(value: Any, arg_type: str, param_info: Dict[str, Any]) -> Any:
        if value is None:
            return None

        if arg_type == "str":
            return str(value)

        if arg_type == "int":
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return value

        if arg_type == "float":
            try:
                return float(value)
            except (TypeError, ValueError):
                return value

        if arg_type == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "y")
            return bool(value)

        if arg_type == "enum":
            options = param_info.get("options", [])
            if not options:
                return value
            value_str = str(value)
            for option in options:
                if str(option).lower() == value_str.lower():
                    return option
            return options[0]

        if arg_type == "List[email]":
            if isinstance(value, list):
                return [str(v) for v in value]
            if isinstance(value, str):
                parts = [part.strip() for part in value.split(",") if part.strip()]
                return parts
            return value

        if arg_type == "DateRange":
            if isinstance(value, dict):
                return value
            return None

        return value

    def _finalize_arguments(
        self,
        request: str,
        action: str,
        tool_name: str,
        model_arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        schema = TOOL_SCHEMAS.get(tool_name, {})
        params = schema.get("parameters", {})
        finalized: Dict[str, Any] = {}

        for arg_name, raw_value in model_arguments.items():
            if arg_name not in params:
                continue
            param_info = params[arg_name]
            arg_type = param_info.get("type", "str")
            finalized[arg_name] = self._coerce_value(raw_value, arg_type, param_info)

        # Fill defaults.
        for arg_name, param_info in params.items():
            if arg_name not in finalized and "default" in param_info:
                finalized[arg_name] = param_info["default"]

        # Fill required arguments using deterministic extraction and fallback.
        for arg_name, param_info in params.items():
            if not param_info.get("required", False):
                continue
            if arg_name in finalized and finalized[arg_name] is not None:
                continue

            arg_type = param_info.get("type", "str")
            extracted = self.arg_fallback.extract_from_query(
                query=action,
                arg_name=arg_name,
                arg_type=arg_type,
                tool_name=tool_name,
            )
            if extracted is None:
                extracted = self.arg_fallback.fallback_value(
                    query=request,
                    arg_name=arg_name,
                    arg_type=arg_type,
                    tool_name=tool_name,
                )
            if extracted is not None:
                finalized[arg_name] = extracted

        return finalized

    def run(
        self,
        request: str,
        max_actions: int = 8,
        similarity_threshold: float = 0.5,
        execute_tools: bool = False,
    ) -> OrchestratorRunResult:
        started = time.perf_counter()
        self.logger.info("run_started request=%s", request)

        atomic_actions = self.planner.plan_atomic_actions(
            request=request,
            max_actions=max_actions,
        )
        self.logger.info("plan_created actions=%s", json.dumps(atomic_actions, ensure_ascii=True))

        steps: List[OrchestratorStepResult] = []
        context_steps: List[Dict[str, Any]] = []

        for action in atomic_actions:
            step_id = int(action["id"])
            instruction = str(action["instruction"])
            depends_on = action.get("depends_on", [])
            self.logger.info(
                "step_started id=%s instruction=%s depends_on=%s",
                step_id,
                instruction,
                depends_on,
            )

            retrieval = self.cluster_system.predict(
                query=instruction,
                top_k=1,
                similarity_threshold=similarity_threshold,
                force_tool_call=False,
            )
            cluster_id = retrieval.cluster_id
            confidence = retrieval.confidence

            if retrieval.tool_name is None or cluster_id < 0:
                step = OrchestratorStepResult(
                    step_id=step_id,
                    action=instruction,
                    cluster_id=cluster_id,
                    confidence=confidence,
                    tool_name=None,
                    arguments={},
                    tool_call=None,
                    status="abstained",
                    validation_errors=["No cluster above threshold"],
                )
                steps.append(step)
                context_steps.append({
                    "id": step_id,
                    "action": instruction,
                    "status": step.status,
                    "tool": None,
                    "result": None,
                })
                self.logger.warning(
                    "step_abstained id=%s cluster_id=%s confidence=%.4f",
                    step_id,
                    cluster_id,
                    confidence,
                )
                continue

            tool_name = retrieval.tool_name
            tool_schema = TOOL_SCHEMAS.get(tool_name, {})
            planner_context = {
                "request": request,
                "depends_on": depends_on,
                "previous_steps": context_steps,
            }
            planner_args = self.planner.infer_arguments(
                action=instruction,
                tool_name=tool_name,
                tool_schema=tool_schema,
                context=planner_context,
            )
            arguments = self._finalize_arguments(
                request=request,
                action=instruction,
                tool_name=tool_name,
                model_arguments=planner_args,
            )

            valid, errors = self.cluster_system.mapper.validate_tool_execution(
                tool_name=tool_name,
                arguments=arguments,
                strict=False,
            )
            tool_call = format_tool_call(tool_name, arguments)
            status = "ready" if valid else "invalid"
            execution: Optional[Dict[str, Any]] = None

            if execute_tools and valid:
                execution = self.cluster_system.mapper.dispatch_cluster(
                    cluster_id=cluster_id,
                    arguments=arguments,
                    similarity_score=confidence,
                    threshold=similarity_threshold,
                    strict=False,
                )
                status = "executed" if execution.get("ok", False) else "execution_failed"

            step = OrchestratorStepResult(
                step_id=step_id,
                action=instruction,
                cluster_id=cluster_id,
                confidence=confidence,
                tool_name=tool_name,
                arguments=arguments,
                tool_call=tool_call,
                status=status,
                validation_errors=errors,
                execution=execution,
            )
            steps.append(step)
            context_steps.append({
                "id": step_id,
                "action": instruction,
                "status": step.status,
                "tool": tool_name,
                "tool_call": tool_call,
                "result": execution.get("result") if execution else None,
            })

            self.logger.info(
                "step_completed id=%s cluster_id=%s tool=%s confidence=%.4f status=%s tool_call=%s",
                step_id,
                cluster_id,
                tool_name,
                confidence,
                status,
                tool_call,
            )
            if errors:
                self.logger.warning("step_validation_errors id=%s errors=%s", step_id, errors)

        duration_ms = (time.perf_counter() - started) * 1000.0
        self.logger.info("run_completed duration_ms=%.3f steps=%s", duration_ms, len(steps))

        return OrchestratorRunResult(
            request=request,
            atomic_actions=atomic_actions,
            steps=steps,
            duration_ms=duration_ms,
        )

from __future__ import annotations

import unittest

from models.software_layer import DispatchResult
from orchestrator.agent import NTILCOrchestratorAgent
from orchestrator.planning import (
    coerce_action_to_natural_language,
    enforce_atomic_actions,
    looks_like_shell_command,
    normalize_actions_to_natural_language,
    parse_plan_block,
)
from orchestrator.results import RetrievalCandidate


class _FakeRetrievalSystem:
    def __init__(self, mapping: dict[str, list[RetrievalCandidate]]):
        self.mapping = mapping

    def retrieve_candidates(self, query: str, top_k: int = 3, threshold: float = -1.0):
        del top_k, threshold
        return list(self.mapping.get(query, []))


class _SequenceDispatcher:
    def __init__(self, sequence: list[DispatchResult]):
        self._sequence = list(sequence)
        self.calls = []

    def dispatch_cluster(self, cluster_id, arguments=None, execute=True, granted_permissions=None):
        self.calls.append({
            "cluster_id": cluster_id,
            "arguments": dict(arguments or {}),
            "execute": execute,
            "granted_permissions": granted_permissions,
        })
        if self._sequence:
            return self._sequence.pop(0)
        return DispatchResult(
            ok=False,
            tool="",
            arguments=dict(arguments or {}),
            cluster_id=cluster_id,
            result=None,
            errors=["No dispatch result queued"],
            executed=execute,
        )


class _FakeQwenModel:
    def __init__(self, plan_block: str):
        self.plan_block = plan_block

    def generate_plan(self, request: str, max_actions: int = 8, max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0):
        del request, max_actions, max_new_tokens, temperature, top_p
        return self.plan_block

    def generate_command(
        self,
        query: str,
        tool: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 1.0,
        current_action=None,
        prior_step_summaries=None,
    ):
        del query, max_new_tokens, temperature, top_p, prior_step_summaries
        action_text = str(current_action or "").strip().replace(" ", "_")
        return {
            "generated_text": f"{tool} --for {action_text}",
            "command": f"{tool} --for {action_text}",
        }


class _FakeProtocolQwenModel:
    def generate_plan_actions(self, request: str, max_actions: int = 8, max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0):
        del request, max_actions, max_new_tokens, temperature, top_p
        return {
            "actions": ["search logs"],
            "plan_block": "<plan><action>search logs</action></plan>",
            "raw_text": "",
            "raw_token_ids": [],
        }

    def generate_dispatch_arguments(
        self,
        query: str,
        tool: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        current_action=None,
        prior_step_summaries=None,
    ):
        del query, max_new_tokens, temperature, top_p, current_action, prior_step_summaries
        return {
            "arguments": {"command": f"{tool} -n error", "query": "search logs"},
            "generated_text": f"{tool} -n error",
            "command": f"{tool} -n error",
            "raw_text": "",
            "raw_token_ids": [],
        }


class PlanningTests(unittest.TestCase):
    def test_parse_plan_block_valid(self):
        block = """
<plan>
  <action><len:14>make directory</len></action>
  <action><len:15>move text files</len></action>
</plan>
"""
        actions = parse_plan_block(block)
        self.assertEqual([a.instruction for a in actions], ["make directory", "move text files"])

    def test_parse_plan_block_invalid_raises(self):
        with self.assertRaises(ValueError):
            parse_plan_block("no plan block")

    def test_atomic_split_on_conjunction(self):
        block = """
<plan>
  <action><len:35>make directory and move text files</len></action>
</plan>
"""
        actions = enforce_atomic_actions(parse_plan_block(block))
        self.assertEqual([a.instruction for a in actions], ["make directory", "move text files"])

    def test_detects_and_rewrites_shell_command_actions(self):
        self.assertTrue(looks_like_shell_command("grep -R cuda ."))
        self.assertEqual(
            coerce_action_to_natural_language("grep -R cuda ."),
            "search for matching text recursively",
        )

    def test_normalizes_shell_command_actions_before_atomic_split(self):
        actions = normalize_actions_to_natural_language(
            parse_plan_block("<plan><action>grep -R cuda .</action></plan>"),
            fallback_request="search recursively for cuda references",
        )
        self.assertEqual([a.instruction for a in actions], ["search for matching text recursively"])


class RuntimeFlowTests(unittest.TestCase):
    def test_retry_then_success_with_single_action(self):
        plan = """
<plan>
  <action><len:11>search logs</len></action>
</plan>
"""
        retrieval = _FakeRetrievalSystem(
            {
                "search logs": [
                    RetrievalCandidate(cluster_id=10, score=0.8, tool_name="grep"),
                    RetrievalCandidate(cluster_id=11, score=0.7, tool_name="find"),
                ]
            }
        )
        dispatcher = _SequenceDispatcher(
            [
                DispatchResult(
                    ok=False,
                    tool="grep",
                    arguments={"command": "grep --for search_logs"},
                    cluster_id=10,
                    result=None,
                    errors=["bad args"],
                    executed=False,
                ),
                DispatchResult(
                    ok=True,
                    tool="find",
                    arguments={"command": "find --for search_logs"},
                    cluster_id=11,
                    result=None,
                    errors=[],
                    executed=False,
                ),
            ]
        )
        model = _FakeQwenModel(plan)

        agent = NTILCOrchestratorAgent(retrieval_system=retrieval, dispatcher=dispatcher, qwen_model=model)
        run = agent.run(request="search logs", execute_tools=False, top_k_candidates=2, max_retries=2)

        self.assertEqual(len(run.atomic_actions), 1)
        self.assertEqual(len(run.steps), 2)
        self.assertFalse(run.steps[0].dispatch_result.ok)
        self.assertTrue(run.steps[1].dispatch_result.ok)
        self.assertTrue(run.success)
        self.assertEqual(run.action_failures, [])

    def test_deterministic_stop_on_unrecoverable_failure(self):
        plan = """
<plan>
  <action><len:24>list files and read readme</len></action>
</plan>
"""
        retrieval = _FakeRetrievalSystem(
            {
                "list files": [RetrievalCandidate(cluster_id=1, score=0.9, tool_name="ls")],
                "read readme": [RetrievalCandidate(cluster_id=2, score=0.9, tool_name="cat")],
            }
        )
        dispatcher = _SequenceDispatcher(
            [
                DispatchResult(
                    ok=False,
                    tool="ls",
                    arguments={"command": "ls --for list_files"},
                    cluster_id=1,
                    result=None,
                    errors=["permission denied"],
                    executed=False,
                )
            ]
        )
        model = _FakeQwenModel(plan)

        agent = NTILCOrchestratorAgent(retrieval_system=retrieval, dispatcher=dispatcher, qwen_model=model)
        run = agent.run(request="list files and read readme", execute_tools=False, max_retries=1)

        self.assertEqual([a["instruction"] for a in run.atomic_actions], ["list files", "read readme"])
        self.assertEqual(len(run.steps), 1)
        self.assertEqual(run.steps[0].action_id, 1)
        self.assertEqual(len(run.action_failures), 1)
        self.assertFalse(run.success)

    def test_protocol_generation_path(self):
        retrieval = _FakeRetrievalSystem(
            {
                "search logs": [RetrievalCandidate(cluster_id=20, score=0.91, tool_name="grep")],
            }
        )
        dispatcher = _SequenceDispatcher(
            [
                DispatchResult(
                    ok=True,
                    tool="grep",
                    arguments={"command": "grep -n error"},
                    cluster_id=20,
                    result=None,
                    errors=[],
                    executed=False,
                )
            ]
        )
        model = _FakeProtocolQwenModel()

        agent = NTILCOrchestratorAgent(retrieval_system=retrieval, dispatcher=dispatcher, qwen_model=model)
        run = agent.run(request="search logs", execute_tools=False, max_retries=1)

        self.assertTrue(run.success)
        self.assertEqual(len(run.steps), 1)
        self.assertEqual(run.steps[0].command, "grep -n error")
        self.assertEqual(run.steps[0].dispatch_arguments.get("query"), "search logs")

    def test_command_like_plan_action_is_normalized_before_retrieval(self):
        retrieval = _FakeRetrievalSystem(
            {
                "search for matching text recursively": [
                    RetrievalCandidate(cluster_id=30, score=0.92, tool_name="grep")
                ],
            }
        )
        dispatcher = _SequenceDispatcher(
            [
                DispatchResult(
                    ok=True,
                    tool="grep",
                    arguments={"command": "grep -R error ."},
                    cluster_id=30,
                    result=None,
                    errors=[],
                    executed=False,
                )
            ]
        )

        class _CommandLikePlanModel:
            def generate_plan_actions(
                self,
                request: str,
                max_actions: int = 8,
                max_new_tokens: int = 256,
                temperature: float = 0.0,
                top_p: float = 1.0,
            ):
                del request, max_actions, max_new_tokens, temperature, top_p
                return {
                    "actions": ["grep -R error ."],
                    "plan_block": "<plan><action>grep -R error .</action></plan>",
                    "raw_text": "",
                    "raw_token_ids": [],
                }

            def generate_dispatch_arguments(
                self,
                query: str,
                tool: str,
                max_new_tokens: int = 128,
                temperature: float = 0.0,
                top_p: float = 1.0,
                current_action=None,
                prior_step_summaries=None,
            ):
                del query, max_new_tokens, temperature, top_p, current_action, prior_step_summaries
                return {
                    "arguments": {"command": f"{tool} -R error .", "query": "search logs"},
                    "generated_text": f"{tool} -R error .",
                    "command": f"{tool} -R error .",
                    "raw_text": "",
                    "raw_token_ids": [],
                }

        agent = NTILCOrchestratorAgent(
            retrieval_system=retrieval,
            dispatcher=dispatcher,
            qwen_model=_CommandLikePlanModel(),
        )
        run = agent.run(request="search logs", execute_tools=False, max_retries=1)

        self.assertTrue(run.success)
        self.assertEqual(
            [a["instruction"] for a in run.atomic_actions],
            ["search for matching text recursively"],
        )
        self.assertEqual(run.steps[0].action_text, "search for matching text recursively")


class CompatibilityTests(unittest.TestCase):
    def test_imports_from_inference_and_orchestrator_agent(self):
        from inference import NTILCOrchestratorAgent as NTILCFromInference
        from inference import OrchestratorRunResult, OrchestratorStepResult
        from orchestrator.agent import NTILCOrchestratorAgent as NTILCFromPackage

        self.assertIs(NTILCFromInference, NTILCFromPackage)
        self.assertTrue(OrchestratorRunResult)
        self.assertTrue(OrchestratorStepResult)

        # These require optional heavy ML deps in this environment.
        try:
            from inference import ClusterBasedToolSystem, QwenOrchestratorModel
        except ModuleNotFoundError as exc:
            if "torch" in str(exc) or "transformers" in str(exc):
                self.skipTest(f"Optional dependency missing for model-backed imports: {exc}")
            raise
        self.assertTrue(ClusterBasedToolSystem)
        self.assertTrue(QwenOrchestratorModel)


if __name__ == "__main__":
    unittest.main()

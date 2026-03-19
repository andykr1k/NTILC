from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    from test import TestInference
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    TestInference = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _IMPORT_ERROR = None


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        return {"input_ids": str(text).split()}


class _FakeRuntimeModel:
    mode = "full"

    def __init__(self):
        self.plan_calls = []
        self.dispatch_calls = []

    def generate_plan_actions(
        self,
        request: str,
        max_actions: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.plan_calls.append(
            {
                "request": request,
                "max_actions": max_actions,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return {
            "actions": ["grep -R cuda ."],
            "plan_block": "<plan><action>grep -R cuda .</action></plan>",
            "raw_text": "<plan><action>grep -R cuda .</action></plan>",
            "raw_token_ids": [11, 12, 13],
        }

    def generate_dispatch_arguments(
        self,
        query: str,
        tool: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 1.0,
        current_action=None,
        prior_step_summaries=None,
    ):
        self.dispatch_calls.append(
            {
                "query": query,
                "tool": tool,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "current_action": current_action,
                "prior_step_summaries": list(prior_step_summaries or []),
            }
        )
        return {
            "arguments": {"command": "grep -n error", "query": query},
            "generated_text": "grep -n error",
            "raw_text": "<dispatch><arg><key>command</key><value>grep -n error</value></arg></dispatch>",
            "raw_token_ids": [21, 22],
        }


def _build_runner(fake_model: _FakeRuntimeModel) -> TestInference:
    runner = TestInference.__new__(TestInference)
    runner.tokenizer = _FakeTokenizer()
    runner.query_encoder = SimpleNamespace(tokenizer=runner.tokenizer)
    runner.qwen_orchestrator_model = fake_model
    runner.max_plan_actions = 8
    runner.plan_max_new_tokens = 256
    runner.max_new_tokens = 96
    runner.temperature = 0.0
    runner.top_p = 1.0
    return runner


@unittest.skipIf(TestInference is None, f"Optional dependency missing for trace runner import: {_IMPORT_ERROR}")
class TraceInferenceTests(unittest.TestCase):
    def test_generate_plan_block_uses_runtime_model(self):
        fake_model = _FakeRuntimeModel()
        runner = _build_runner(fake_model)

        plan = runner.generate_plan_block("search recursively for cuda references")

        self.assertEqual(len(fake_model.plan_calls), 1)
        self.assertEqual(plan["actions"], ["search for matching text recursively"])
        self.assertEqual(plan["metrics"]["generated_tokens"], 3)
        self.assertIn("<plan>", plan["plan_block"])
        self.assertTrue(plan["prompt"].startswith("You are a Linux tool-use planner."))

    def test_generate_dispatch_block_uses_runtime_model(self):
        fake_model = _FakeRuntimeModel()
        runner = _build_runner(fake_model)

        dispatch = runner.generate_dispatch_block(
            plan={
                "request": "search logs",
                "action": "search logs",
                "prior_step_summaries": ["action#1 list files -> ls [ok]"],
            },
            tool_result={"tool_name": "grep", "cluster_id": 7},
        )

        self.assertEqual(len(fake_model.dispatch_calls), 1)
        self.assertEqual(dispatch["command"], "grep -n error")
        self.assertEqual(dispatch["dispatch_arguments"]["query"], "search logs")
        self.assertEqual(dispatch["metrics"]["generated_tokens"], 2)
        self.assertIn("<key>command</key>", dispatch["dispatch_block"])


if __name__ == "__main__":
    unittest.main()

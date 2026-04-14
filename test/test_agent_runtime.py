from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from agent.runtime import AgentController, RuntimeConfig, ToolCandidate, ToolRetriever


class FakeModelAdapter:
    def __init__(self, assistant_outputs: list[str], dispatch_outputs: list[dict] | None = None) -> None:
        self.assistant_outputs = list(assistant_outputs)
        self.dispatch_outputs = list(dispatch_outputs or [])
        self.seen_transcripts: list[list[dict[str, str]]] = []

    def stream_assistant(self, transcript):
        self.seen_transcripts.append([dict(message) for message in transcript])
        output = self.assistant_outputs.pop(0)
        for index in range(0, len(output), 6):
            yield output[index : index + 6]

    def generate_dispatch_arguments(self, transcript, user_message, tool_name, schema):
        if not self.dispatch_outputs:
            raise RuntimeError("No dispatch payload queued.")
        payload = self.dispatch_outputs.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload

    def count_text_tokens(self, text: str) -> int:
        return len(str(text).split())

    def count_assistant_prompt_tokens(self, transcript) -> int:
        return sum(self.count_text_tokens(message["content"]) for message in transcript)

    def count_dispatch_prompt_tokens(self, transcript, user_message, tool_name, schema) -> int:
        return (
            self.count_assistant_prompt_tokens(transcript)
            + self.count_text_tokens(user_message)
            + self.count_text_tokens(tool_name)
        )


class FakeRetriever:
    def __init__(self, mapping: dict[str, list[ToolCandidate]]) -> None:
        self.mapping = mapping
        self.queries: list[str] = []

    def query(self, query_text: str, top_k: int = 5) -> list[ToolCandidate]:
        self.queries.append(query_text)
        return self.mapping.get(query_text, [])[:top_k]


class TestToolRetriever(unittest.TestCase):
    def setUp(self) -> None:
        self.tool_registry = [
            {"name": "alpha", "description": "Alpha tool", "category": "test", "parameters": {}},
            {"name": "beta", "description": "Beta tool", "category": "test", "parameters": {}},
        ]

    def _temp_checkpoint_path(self) -> Path:
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            return Path(handle.name)

    def test_missing_checkpoint_fails_fast(self) -> None:
        missing_path = Path(tempfile.gettempdir()) / "definitely-missing-tool-checkpoint.pt"
        if missing_path.exists():
            missing_path.unlink()
        with self.assertRaises(RuntimeError) as ctx:
            ToolRetriever.from_checkpoint(
                missing_path,
                self.tool_registry,
                device="cpu",
                batch_size=4,
            )
        self.assertIn("Embedding checkpoint not found", str(ctx.exception))

    def test_catalog_mismatch_raises_error(self) -> None:
        checkpoint_path = self._temp_checkpoint_path()
        bundle = {
            "model": object(),
            "tokenizer": object(),
            "max_length": 32,
            "tool_names": ["alpha", "gamma"],
            "centroids": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        with patch("agent.runtime.load_checkpoint_bundle", return_value=bundle):
            with self.assertRaises(RuntimeError) as ctx:
                ToolRetriever.from_checkpoint(
                    checkpoint_path,
                    self.tool_registry,
                    device="cpu",
                    batch_size=4,
                )
        self.assertIn("do not match", str(ctx.exception))

    def test_query_returns_ranked_candidates(self) -> None:
        checkpoint_path = self._temp_checkpoint_path()
        bundle = {
            "model": object(),
            "tokenizer": object(),
            "max_length": 32,
            "tool_names": ["alpha", "beta"],
            "centroids": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        }
        with patch("agent.runtime.load_checkpoint_bundle", return_value=bundle), patch(
            "agent.runtime.embed_texts",
            return_value=torch.tensor([[0.9, 0.1]], dtype=torch.float32),
        ):
            retriever = ToolRetriever.from_checkpoint(
                checkpoint_path,
                self.tool_registry,
                device="cpu",
                batch_size=4,
            )
            candidates = retriever.query("find alpha", top_k=1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "alpha")
        self.assertEqual(candidates[0].description, "Alpha tool")


class TestAgentController(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RuntimeConfig(top_k=3, max_tool_steps=3, max_agent_passes=6)
        self.tool_by_name = {
            "calculate": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression.",
                "category": "computation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                    },
                    "required": ["expression"],
                },
            }
        }

    def test_direct_answer_without_tools(self) -> None:
        controller = AgentController(
            config=self.config,
            model_adapter=FakeModelAdapter(["This can be answered directly."]),
            retriever=FakeRetriever({}),
            tool_by_name=self.tool_by_name,
            executor=lambda *_args, **_kwargs: {},
        )
        result = controller.run_turn(user_message="What is 2+2?", transcript=[])
        self.assertEqual(result.assistant_text, "This can be answered directly.")
        self.assertEqual(result.events, [])
        self.assertEqual(result.transcript[-1]["content"], "This can be answered directly.")
        self.assertGreater(result.stats["assistant_output_tokens"], 0)
        self.assertEqual(result.stats["tool_call_count"], 0)

    def test_search_select_dispatch_and_final_answer(self) -> None:
        model = FakeModelAdapter(
            [
                "I'll use a tool.\n<search_tools>calculate a simple expression</search_tools>",
                "<select_tool>calculate</select_tool>",
                "The result is 4.",
            ],
            dispatch_outputs=[{"expression": "2+2"}],
        )
        retriever = FakeRetriever(
            {
                "calculate a simple expression": [
                    ToolCandidate(
                        name="calculate",
                        description="Evaluate a mathematical expression.",
                        category="computation",
                        score=0.98,
                    )
                ]
            }
        )

        def executor(tool_name, arguments, tool_spec):
            return {
                "tool": tool_name,
                "arguments": arguments,
                "status": "ok",
                "output": {"result": 4},
                "error": None,
            }

        controller = AgentController(
            config=self.config,
            model_adapter=model,
            retriever=retriever,
            tool_by_name=self.tool_by_name,
            executor=executor,
        )
        result = controller.run_turn(user_message="What is 2+2?", transcript=[])
        self.assertIn("I'll use a tool.", result.assistant_text)
        self.assertIn("The result is 4.", result.assistant_text)
        self.assertEqual([event["type"] for event in result.events], ["search", "dispatch", "response"])
        self.assertEqual(result.events[-1]["status"], "ok")
        self.assertTrue(any("<dispatch>" in item["content"] for item in result.transcript if item["role"] == "user"))
        self.assertEqual(result.stats["search_count"], 1)
        self.assertEqual(result.stats["dispatch_count"], 1)
        self.assertEqual(result.stats["tool_success_count"], 1)
        self.assertIn("calculate", result.stats["tools_used"])
        self.assertGreater(result.stats["model_input_tokens"], 0)
        self.assertGreater(result.stats["model_output_tokens"], 0)

    def test_failed_tool_execution_allows_graceful_answer(self) -> None:
        model = FakeModelAdapter(
            [
                "<search_tools>calculate a broken expression</search_tools>",
                "<select_tool>calculate</select_tool>",
                "The calculator failed, so I cannot complete the request.",
            ],
            dispatch_outputs=[{"expression": "bad("}],
        )
        retriever = FakeRetriever(
            {
                "calculate a broken expression": [
                    ToolCandidate(
                        name="calculate",
                        description="Evaluate a mathematical expression.",
                        category="computation",
                        score=0.95,
                    )
                ]
            }
        )

        def executor(tool_name, arguments, tool_spec):
            return {
                "tool": tool_name,
                "arguments": arguments,
                "status": "error",
                "output": None,
                "error": "Parser failure inside calculate",
            }

        controller = AgentController(
            config=self.config,
            model_adapter=model,
            retriever=retriever,
            tool_by_name=self.tool_by_name,
            executor=executor,
        )
        result = controller.run_turn(user_message="Break the calculator.", transcript=[])
        self.assertEqual(result.events[-1]["type"], "response")
        self.assertEqual(result.events[-1]["status"], "error")
        self.assertIn("failed", result.assistant_text.lower())
        self.assertEqual(result.stats["tool_error_count"], 1)

    def test_multi_turn_transcript_carries_prior_response_context(self) -> None:
        model = FakeModelAdapter(
            [
                "<search_tools>calculate 2 plus 2</search_tools>",
                "<select_tool>calculate</select_tool>",
                "The result is 4.",
                "Using the earlier tool result, the recap is still 4.",
            ],
            dispatch_outputs=[{"expression": "2+2"}],
        )
        retriever = FakeRetriever(
            {
                "calculate 2 plus 2": [
                    ToolCandidate(
                        name="calculate",
                        description="Evaluate a mathematical expression.",
                        category="computation",
                        score=0.99,
                    )
                ]
            }
        )

        def executor(tool_name, arguments, tool_spec):
            return {
                "tool": tool_name,
                "arguments": arguments,
                "status": "ok",
                "output": {"result": 4},
                "error": None,
            }

        controller = AgentController(
            config=self.config,
            model_adapter=model,
            retriever=retriever,
            tool_by_name=self.tool_by_name,
            executor=executor,
        )
        first_turn = controller.run_turn(user_message="What is 2+2?", transcript=[])
        second_turn = controller.run_turn(
            user_message="Summarize the earlier answer.",
            transcript=first_turn.transcript,
        )
        self.assertIn("recap is still 4", second_turn.assistant_text)
        self.assertGreaterEqual(len(model.seen_transcripts), 4)
        latest_transcript = model.seen_transcripts[-1]
        self.assertTrue(any("<response>" in message["content"] for message in latest_transcript))
        self.assertGreater(second_turn.stats["assistant_prompt_tokens"], 0)


if __name__ == "__main__":
    unittest.main()

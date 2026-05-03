import io
import json
import os
import urllib.error
import unittest
from unittest.mock import patch

from benchmark.adapters import (
    GeminiSelectionAdapter,
    OpenAISelectionAdapter,
    build_api_selection_tool,
    build_model_summary,
    extract_anthropic_tool_arguments,
    extract_gemini_tool_arguments,
    extract_openai_tool_arguments,
    http_post_json,
    is_cuda_oom_error,
    is_rate_limit_response,
)


class BenchmarkAdapterStructuredOutputTests(unittest.TestCase):
    def test_build_model_summary_marks_all_error_runs_as_error(self) -> None:
        summary = build_model_summary(
            adapter_id="hf/test-model",
            provider="huggingface",
            mode="llm_local",
            model_name="test-model",
            results=[
                {
                    "status": "error",
                    "error_message": "CUDA out of memory.",
                }
            ],
        )
        self.assertEqual(summary["status"], "error")
        self.assertEqual(summary["error_message"], "CUDA out of memory.")
        self.assertEqual(summary["metrics"]["successful_examples"], 0)

    def test_build_api_selection_tool_uses_enum_schema(self) -> None:
        selection_tool = build_api_selection_tool(
            ["weather", "stocks", "search"],
            ranking_limit=3,
        )
        self.assertEqual(selection_tool["name"], "select_tool")
        properties = selection_tool["parameters"]["properties"]
        self.assertEqual(properties["selected_tool"]["enum"], ["weather", "stocks", "search"])

    def test_is_cuda_oom_error_detects_oom_message(self) -> None:
        self.assertTrue(is_cuda_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")))
        self.assertFalse(is_cuda_oom_error(RuntimeError("some other runtime error")))

    def test_is_rate_limit_response_detects_provider_errors(self) -> None:
        self.assertTrue(is_rate_limit_response(429, '{"error":{"type":"rate_limit_error"}}'))
        self.assertTrue(is_rate_limit_response(400, "rate limit exceeded"))
        self.assertFalse(is_rate_limit_response(404, "model not found"))

    def test_http_post_json_retries_once_after_rate_limit(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return b'{"ok": true}'

        rate_limit_error = urllib.error.HTTPError(
            url="https://api.example.test/v1",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(b'{"error":{"type":"rate_limit_error"}}'),
        )

        with patch("benchmark.adapters.urllib.request.urlopen", side_effect=[rate_limit_error, FakeResponse()]) as urlopen:
            with patch("benchmark.adapters.time.sleep") as sleep:
                payload = http_post_json(
                    url="https://api.example.test/v1",
                    headers={},
                    payload={"model": "test"},
                    timeout_seconds=1,
                )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(20)

    def test_extract_openai_tool_arguments_reads_tool_call(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "select_tool",
                                    "arguments": '{"selected_tool":"weather","ranked_tools":["weather"],"reason":"best"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        self.assertEqual(
            extract_openai_tool_arguments(response),
            '{"selected_tool":"weather","ranked_tools":["weather"],"reason":"best"}',
        )

    def test_extract_anthropic_tool_arguments_reads_tool_use_block(self) -> None:
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "name": "select_tool",
                    "input": {
                        "selected_tool": "stocks",
                        "ranked_tools": ["stocks"],
                        "reason": "best",
                    },
                }
            ]
        }
        payload = json.loads(extract_anthropic_tool_arguments(response))
        self.assertEqual(payload["selected_tool"], "stocks")
        self.assertEqual(payload["ranked_tools"], ["stocks"])

    def test_extract_gemini_tool_arguments_reads_function_call(self) -> None:
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "select_tool",
                                    "args": {
                                        "selected_tool": "search",
                                        "ranked_tools": ["search"],
                                        "reason": "best",
                                    },
                                }
                            }
                        ]
                    }
                }
            ]
        }
        payload = json.loads(extract_gemini_tool_arguments(response))
        self.assertEqual(payload["selected_tool"], "search")
        self.assertEqual(payload["ranked_tools"], ["search"])

    def test_openai_omits_output_cap_by_default(self) -> None:
        previous_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        captured_payload = {}

        def fake_post_json(**kwargs):
            captured_payload.update(kwargs["payload"])
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "select_tool",
                                        "arguments": (
                                            '{"selected_tool":"weather",'
                                            '"ranked_tools":["weather"],'
                                            '"reason":"best"}'
                                        ),
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usage": {},
            }

        try:
            with patch("benchmark.adapters.http_post_json", side_effect=fake_post_json):
                adapter = OpenAISelectionAdapter(
                    "gpt-test",
                    ranking_limit=1,
                    max_output_tokens=None,
                    timeout_seconds=1,
                    pricing=None,
                )
                adapter.call_api(
                    [{"role": "user", "content": "weather"}],
                    valid_tool_names=["weather"],
                )
        finally:
            if previous_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous_key

        self.assertNotIn("max_tokens", captured_payload)
        self.assertNotIn("max_completion_tokens", captured_payload)

    def test_openai_explicit_output_cap_uses_max_completion_tokens(self) -> None:
        previous_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        captured_payload = {}

        def fake_post_json(**kwargs):
            captured_payload.update(kwargs["payload"])
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "select_tool",
                                        "arguments": (
                                            '{"selected_tool":"weather",'
                                            '"ranked_tools":["weather"],'
                                            '"reason":"best"}'
                                        ),
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usage": {},
            }

        try:
            with patch("benchmark.adapters.http_post_json", side_effect=fake_post_json):
                adapter = OpenAISelectionAdapter(
                    "gpt-test",
                    ranking_limit=1,
                    max_output_tokens=32,
                    timeout_seconds=1,
                    pricing=None,
                )
                adapter.call_api(
                    [{"role": "user", "content": "weather"}],
                    valid_tool_names=["weather"],
                )
        finally:
            if previous_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous_key

        self.assertNotIn("max_tokens", captured_payload)
        self.assertEqual(captured_payload["max_completion_tokens"], 32)

    def test_gemini_omits_output_cap_by_default(self) -> None:
        previous_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "test-key"
        captured_payload = {}

        def fake_post_json(**kwargs):
            captured_payload.update(kwargs["payload"])
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '{"selected_tool":"weather",'
                                        '"ranked_tools":["weather"],'
                                        '"reason":"best"}'
                                    )
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {},
            }

        try:
            with patch("benchmark.adapters.http_post_json", side_effect=fake_post_json):
                adapter = GeminiSelectionAdapter(
                    "gemini-test",
                    ranking_limit=1,
                    max_output_tokens=None,
                    timeout_seconds=1,
                    pricing=None,
                )
                adapter.call_api(
                    [{"role": "user", "content": "weather"}],
                    valid_tool_names=["weather"],
                )
        finally:
            if previous_key is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = previous_key

        self.assertNotIn("maxOutputTokens", captured_payload["generationConfig"])


if __name__ == "__main__":
    unittest.main()

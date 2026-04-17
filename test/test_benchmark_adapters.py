import json
import unittest

from benchmark.adapters import (
    build_api_selection_tool,
    extract_anthropic_tool_arguments,
    extract_gemini_tool_arguments,
    extract_openai_tool_arguments,
)


class BenchmarkAdapterStructuredOutputTests(unittest.TestCase):
    def test_build_api_selection_tool_uses_enum_schema(self) -> None:
        selection_tool = build_api_selection_tool(
            ["weather", "stocks", "search"],
            ranking_limit=3,
        )
        self.assertEqual(selection_tool["name"], "select_tool")
        properties = selection_tool["parameters"]["properties"]
        self.assertEqual(properties["selected_tool"]["enum"], ["weather", "stocks", "search"])

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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import patch

from REPL.tools import dispatch_tool_call


class TestDispatchToolCall(unittest.TestCase):
    def test_applies_defaults_before_execution(self) -> None:
        tool_spec = {
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "default": "2+2"},
                },
                "required": [],
            }
        }
        with patch("REPL.tools.execute_tool", return_value={"result": 4}) as execute_mock:
            result = dispatch_tool_call("calculate", {}, tool_spec)
        execute_mock.assert_called_once_with("calculate", {"expression": "2+2"})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["arguments"]["expression"], "2+2")

    def test_rejects_missing_required_arguments(self) -> None:
        tool_spec = {
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            }
        }
        result = dispatch_tool_call("web_search", {}, tool_spec)
        self.assertEqual(result["status"], "error")
        self.assertIn("Missing required arguments", result["error"])

    def test_rejects_unknown_tools(self) -> None:
        result = dispatch_tool_call("definitely_missing_tool", {}, None)
        self.assertEqual(result["status"], "error")
        self.assertIn("Unknown tool", result["error"])

    def test_rejects_extra_arguments(self) -> None:
        tool_spec = {
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": [],
            }
        }
        result = dispatch_tool_call(
            "calculate",
            {"expression": "2+2", "unexpected": True},
            tool_spec,
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("Unexpected arguments", result["error"])

    def test_rejects_type_and_enum_violations(self) -> None:
        type_spec = {
            "parameters": {
                "type": "object",
                "properties": {
                    "recursive": {"type": "boolean"},
                },
                "required": [],
            }
        }
        enum_spec = {
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {"type": "string", "enum": ["text", "markdown", "html"]},
                },
                "required": [],
            }
        }
        type_result = dispatch_tool_call("list_directory", {"recursive": "yes"}, type_spec)
        enum_result = dispatch_tool_call("fetch_url", {"format": "pdf"}, enum_spec)
        self.assertEqual(type_result["status"], "error")
        self.assertIn("schema type", type_result["error"])
        self.assertEqual(enum_result["status"], "error")
        self.assertIn("must be one of", enum_result["error"])

    def test_wraps_unexpected_execution_failures(self) -> None:
        tool_spec = {
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            }
        }
        with patch("REPL.tools.execute_tool", side_effect=RuntimeError("boom")):
            result = dispatch_tool_call("calculate", {"expression": "2+2"}, tool_spec)
        self.assertEqual(result["status"], "error")
        self.assertIn("Unexpected tool execution failure", result["error"])


if __name__ == "__main__":
    unittest.main()

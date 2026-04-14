from __future__ import annotations

import unittest

from agent.protocol import (
    ASSISTANT_CONTROL_TAGS,
    CONTROLLER_CONTROL_TAGS,
    decode_json_block,
    parse_text_with_control_blocks,
)


class TestAgentProtocol(unittest.TestCase):
    def test_plain_text_without_control_block(self) -> None:
        snapshot = parse_text_with_control_blocks("Hello from the assistant.", ASSISTANT_CONTROL_TAGS)
        self.assertEqual(snapshot.visible_text, "Hello from the assistant.")
        self.assertIsNone(snapshot.block)
        self.assertIsNone(snapshot.issue)
        self.assertEqual(snapshot.trimmed_output, "Hello from the assistant.")

    def test_complete_control_block_with_visible_prefix(self) -> None:
        snapshot = parse_text_with_control_blocks(
            "Looking up a tool.\n<search_tools>I need to create a file</search_tools>",
            ASSISTANT_CONTROL_TAGS,
        )
        self.assertEqual(snapshot.visible_text, "Looking up a tool.\n")
        self.assertIsNotNone(snapshot.block)
        assert snapshot.block is not None
        self.assertEqual(snapshot.block.tag, "search_tools")
        self.assertEqual(snapshot.block.content, "I need to create a file")
        self.assertIsNone(snapshot.issue)

    def test_incomplete_control_block_reports_issue(self) -> None:
        snapshot = parse_text_with_control_blocks(
            "Working on it <search_",
            ASSISTANT_CONTROL_TAGS,
        )
        self.assertEqual(snapshot.visible_text, "Working on it ")
        self.assertIsNone(snapshot.block)
        self.assertIsNotNone(snapshot.issue)
        assert snapshot.issue is not None
        self.assertIn("Incomplete", snapshot.issue.message)

    def test_malformed_json_inside_controller_block(self) -> None:
        snapshot = parse_text_with_control_blocks(
            '<dispatch>{"tool": "calculate", "arguments": }</dispatch>',
            CONTROLLER_CONTROL_TAGS,
        )
        self.assertIsNotNone(snapshot.block)
        assert snapshot.block is not None
        payload, error = decode_json_block(snapshot.block)
        self.assertIsNone(payload)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertIn("Malformed JSON", error)

    def test_mixed_text_and_control_block(self) -> None:
        snapshot = parse_text_with_control_blocks(
            "I need a tool.\n<select_tool>calculate</select_tool>",
            ASSISTANT_CONTROL_TAGS,
        )
        self.assertEqual(snapshot.visible_text, "I need a tool.\n")
        self.assertIsNotNone(snapshot.block)
        assert snapshot.block is not None
        self.assertEqual(snapshot.block.tag, "select_tool")
        self.assertEqual(snapshot.block.content, "calculate")


if __name__ == "__main__":
    unittest.main()

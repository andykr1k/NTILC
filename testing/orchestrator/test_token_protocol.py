from __future__ import annotations

import unittest

from orchestrator.protocol import (
    ACTION_END,
    ACTION_START,
    ARG_END,
    ARG_START,
    DISPATCH_END,
    DISPATCH_START,
    KEY_END,
    KEY_START,
    PLAN_END,
    PLAN_START,
    VALUE_END,
    VALUE_START,
    extract_dispatch_arguments_from_ids,
    extract_plan_actions_from_ids,
    resolve_protocol_token_ids,
)


class _FakeTokenizer:
    def __init__(self):
        self._vocab = {
            PLAN_START: 1,
            PLAN_END: 2,
            ACTION_START: 3,
            ACTION_END: 4,
            DISPATCH_START: 5,
            DISPATCH_END: 6,
            ARG_START: 7,
            ARG_END: 8,
            KEY_START: 9,
            KEY_END: 10,
            VALUE_START: 11,
            VALUE_END: 12,
            "make": 20,
            "directory": 21,
            "move": 22,
            "files": 23,
            "command": 24,
            "query": 25,
            "mkdir": 26,
            "docs": 27,
            "user": 28,
            "request": 29,
        }
        self._inv = {v: k for k, v in self._vocab.items()}
        self.unk_token_id = 0

    def convert_tokens_to_ids(self, token: str):
        return self._vocab.get(token, self.unk_token_id)

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(self._inv.get(int(i), "") for i in token_ids).strip()


class TokenProtocolTests(unittest.TestCase):
    def test_extract_plan_actions_from_ids(self):
        tok = _FakeTokenizer()
        ids = resolve_protocol_token_ids(tok, strict=True)
        generated = [
            ids.plan_start,
            ids.action_start,
            20,
            21,
            ids.action_end,
            ids.action_start,
            22,
            23,
            ids.action_end,
            ids.plan_end,
        ]
        actions = extract_plan_actions_from_ids(tok, generated, ids)
        self.assertEqual(actions, ["make directory", "move files"])

    def test_extract_dispatch_arguments_from_ids(self):
        tok = _FakeTokenizer()
        ids = resolve_protocol_token_ids(tok, strict=True)
        generated = [
            ids.dispatch_start,
            ids.arg_start,
            ids.key_start,
            24,
            ids.key_end,
            ids.value_start,
            26,
            27,
            ids.value_end,
            ids.arg_end,
            ids.arg_start,
            ids.key_start,
            25,
            ids.key_end,
            ids.value_start,
            28,
            29,
            ids.value_end,
            ids.arg_end,
            ids.dispatch_end,
        ]

        args = extract_dispatch_arguments_from_ids(tok, generated, ids)
        self.assertEqual(args.get("command"), "mkdir docs")
        self.assertEqual(args.get("query"), "user request")


if __name__ == "__main__":
    unittest.main()

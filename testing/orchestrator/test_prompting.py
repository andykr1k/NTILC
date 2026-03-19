from __future__ import annotations

import unittest

from orchestrator.generation.prompting import (
    build_dispatch_model_input,
    build_plan_model_input,
    build_plan_prompt,
    render_chat_prompt,
)


class _ChatTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    ):
        self.calls.append(
            {
                "messages": list(messages),
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        return "<|im_start|>assistant\n"


class _LegacyChatTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.calls.append(
            {
                "messages": list(messages),
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "legacy-chat-prompt"


class PromptingTests(unittest.TestCase):
    def test_render_chat_prompt_uses_chat_template_when_available(self):
        tokenizer = _ChatTokenizer()

        prompt = build_plan_model_input(
            tokenizer=tokenizer,
            request="search recursively for cuda references",
            max_actions=4,
        )

        self.assertEqual(prompt, "<|im_start|>assistant\n")
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertFalse(tokenizer.calls[0]["tokenize"])
        self.assertTrue(tokenizer.calls[0]["add_generation_prompt"])
        self.assertFalse(tokenizer.calls[0]["enable_thinking"])
        self.assertEqual(tokenizer.calls[0]["messages"][0]["role"], "system")
        self.assertEqual(tokenizer.calls[0]["messages"][1]["role"], "user")

    def test_render_chat_prompt_retries_without_enable_thinking_for_legacy_tokenizers(self):
        tokenizer = _LegacyChatTokenizer()

        prompt = build_dispatch_model_input(
            tokenizer=tokenizer,
            query="search logs",
            tool="grep",
            mode="full",
            current_action="search logs",
            prior_step_summaries=["action#1 list files -> ls [ok]"],
        )

        self.assertEqual(prompt, "legacy-chat-prompt")
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertTrue(tokenizer.calls[0]["add_generation_prompt"])

    def test_render_chat_prompt_falls_back_to_raw_prompt_without_chat_template(self):
        fallback_prompt = build_plan_prompt(request="list files", max_actions=3)
        prompt = render_chat_prompt(
            tokenizer=object(),
            messages=[{"role": "user", "content": "ignored"}],
            fallback_prompt=fallback_prompt,
            enable_thinking=False,
        )

        self.assertEqual(prompt, fallback_prompt)
        self.assertIn("Linux task planner", prompt)


if __name__ == "__main__":
    unittest.main()

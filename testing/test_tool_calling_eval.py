from __future__ import annotations

import unittest

from evaluation.metrics import (
    TRAIN_OVERLAP_SPLIT,
    UNSEEN_ONLY_SPLIT,
    aggregate_predictions,
    build_tool_metadata,
    canonicalize_command,
    is_complex_shell_command,
    prepare_eval_rows,
    select_eval_rows,
)
from evaluation.prompt_baseline import parse_strict_json_tool_call


class EvalMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_tools = [
            {
                "name": "grep",
                "one_line": "search for patterns",
                "invocation": "grep [OPTION]... PATTERNS [FILE]...",
            },
            {
                "name": "babeltrace2-run",
                "one_line": "run a Babeltrace 2 graph",
                "invocation": "babeltrace2 [GENERAL OPTIONS] run [--retry-duration=TIME-US] --connect=CONN-RULE ... COMPONENTS",
            },
            {
                "name": "autofsd-probe",
                "one_line": "probe autofsd",
                "invocation": "$PCP_BINADM_DIR/autofsd-probe [host] [timeout]",
            },
        ]
        self.tool_metadata = build_tool_metadata(self.raw_tools)

    def test_prepare_eval_rows_marks_overlap_and_unseen(self) -> None:
        clean_rows = [
            {"tool": "grep", "nl_query": "find errors", "command": "grep error log.txt"},
            {"tool": "grep", "nl_query": "find warnings", "command": "grep warning app.log"},
        ]
        train_rows = [
            {"tool": "grep", "nl_query": "find errors", "command": "grep error log.txt"},
        ]

        prepared = prepare_eval_rows(clean_rows, train_rows, self.tool_metadata)
        self.assertEqual(prepared[0]["split"], TRAIN_OVERLAP_SPLIT)
        self.assertEqual(prepared[1]["split"], UNSEEN_ONLY_SPLIT)

        unseen_only = select_eval_rows(
            prepared,
            split=UNSEEN_ONLY_SPLIT,
            include_complex=False,
            num_samples=None,
            seed=42,
        )
        self.assertEqual(len(unseen_only), 1)
        self.assertEqual(unseen_only[0]["nl_query"], "find warnings")

    def test_complex_shell_detection_filters_only_shell_composition(self) -> None:
        self.assertFalse(is_complex_shell_command("grep error log.txt"))
        self.assertFalse(is_complex_shell_command("comm --output-delimiter='>' file1.txt file2.txt"))
        self.assertTrue(is_complex_shell_command("echo hello | base64"))
        self.assertTrue(is_complex_shell_command("false && echo ok"))
        self.assertTrue(is_complex_shell_command("cat README.md > out.txt"))
        self.assertTrue(is_complex_shell_command("cronnext -t $(date +%s)"))

    def test_canonicalize_command_handles_alias_and_env_prefixed_invocations(self) -> None:
        alias_command = canonicalize_command(
            "babeltrace2-run --connect=src:out:sink",
            tool_name="babeltrace2-run",
            tool_metadata=self.tool_metadata,
        )
        env_prefixed_command = canonicalize_command(
            "$PCP_BINADM_DIR/autofsd-probe host1",
            tool_name="autofsd-probe",
            tool_metadata=self.tool_metadata,
        )

        self.assertEqual(alias_command, "babeltrace2 run --connect=src:out:sink")
        self.assertEqual(env_prefixed_command, "autofsd-probe host1")

    def test_parse_strict_json_tool_call_rejects_malformed_output(self) -> None:
        valid = parse_strict_json_tool_call('{"tool":"grep","command":"grep error log.txt"}')
        trailing = parse_strict_json_tool_call('{"tool":"grep","command":"grep error log.txt"} extra')
        missing_key = parse_strict_json_tool_call('{"tool":"grep"}')

        self.assertTrue(valid["parse_ok"])
        self.assertEqual(valid["tool"], "grep")
        self.assertFalse(trailing["parse_ok"])
        self.assertFalse(missing_key["parse_ok"])

    def test_aggregate_predictions_summarizes_boolean_metrics(self) -> None:
        predictions = [
            {
                "ntilc": {
                    "latency_seconds": 1.0,
                    "raw_exact_match": True,
                    "canonical_exact_match": True,
                    "command_tool_match": True,
                    "structured_output": True,
                    "retrieval_top1_label_match": True,
                    "retrieval_hit_at_k": True,
                },
                "baseline": {
                    "latency_seconds": 2.0,
                    "raw_exact_match": False,
                    "canonical_exact_match": False,
                    "command_tool_match": True,
                    "strict_json_parse": True,
                },
            },
            {
                "ntilc": {
                    "latency_seconds": 3.0,
                    "raw_exact_match": False,
                    "canonical_exact_match": True,
                    "command_tool_match": False,
                    "structured_output": True,
                    "retrieval_top1_label_match": False,
                    "retrieval_hit_at_k": True,
                },
                "baseline": {
                    "latency_seconds": 4.0,
                    "raw_exact_match": True,
                    "canonical_exact_match": True,
                    "command_tool_match": True,
                    "strict_json_parse": False,
                },
            },
        ]

        summary = aggregate_predictions(predictions)
        self.assertEqual(summary["num_examples"], 2)
        self.assertAlmostEqual(summary["systems"]["ntilc"]["canonical_exact_match"], 1.0)
        self.assertAlmostEqual(summary["systems"]["ntilc"]["retrieval_top1_label_match"], 0.5)
        self.assertAlmostEqual(summary["systems"]["baseline"]["strict_json_parse_rate"], 0.5)
        self.assertAlmostEqual(summary["systems"]["baseline"]["raw_exact_match"], 0.5)


if __name__ == "__main__":
    unittest.main()

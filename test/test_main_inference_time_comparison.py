import tempfile
import unittest
from pathlib import Path

from benchmark.main_inference_time_comparison import (
    MethodSpec,
    build_failure_matrix_rows,
    build_failure_rows,
    build_latex_table,
    infer_embedding_variant,
    partition_methods,
)


class MainInferenceTimeComparisonTests(unittest.TestCase):
    def test_infer_embedding_variant_preserves_nested_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "MetaTool"
            checkpoint = dataset_dir / "output" / "normal" / "functional_margin" / "compatibility_weight_1" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")

            variant = infer_embedding_variant(dataset_dir, checkpoint)

        self.assertEqual(variant.variant_id, "normal/functional_margin/compatibility_weight_1")
        self.assertEqual(variant.architecture, "normal")
        self.assertEqual(variant.loss_name, "functional_margin")

    def test_latex_table_uses_seven_column_tabular(self) -> None:
        method = MethodSpec(
            key="ntilc",
            label="NTILC",
            provider="embedding",
            mode="embedding",
            model_name="normal/functional_margin",
        )
        table = build_latex_table(
            [
                {
                    "dataset": "MetaTool",
                    "method_key": method.key,
                    "method": method.label,
                    "registry_tokens": 0,
                    "total_tokens": 16.4,
                    "top_1_accuracy": 0.9,
                    "top_5_accuracy": 1.0,
                    "latency_ms": 12.2,
                }
            ],
            dataset_order=["MetaTool"],
            method_order=[method.key],
        )

        self.assertIn(r"\begin{tabular}{lllcccc}", table)
        self.assertNotIn(r"\begin{tabular}{lllccccc}", table)
        self.assertIn(r"\textbf{90.00\%}", table)

    def test_partition_methods_preserves_requested_order(self) -> None:
        methods = [
            MethodSpec("qwen_ict", "Qwen", "huggingface", "llm_local", "qwen"),
            MethodSpec("openai_ict", "OpenAI", "openai", "llm_api", "gpt"),
            MethodSpec("ntilc", "NTILC", "embedding", "embedding", "ntilc"),
            MethodSpec("gemini_ict", "Gemini", "gemini", "llm_api", "gemini"),
            MethodSpec("anthropic_ict", "Claude", "anthropic", "llm_api", "claude"),
        ]

        api_methods, local_methods = partition_methods(methods)

        self.assertEqual([method.key for method in api_methods], ["openai_ict", "gemini_ict", "anthropic_ict"])
        self.assertEqual([method.key for method in local_methods], ["qwen_ict", "ntilc"])
        self.assertEqual(
            [method.key for method in methods],
            ["qwen_ict", "openai_ict", "ntilc", "gemini_ict", "anthropic_ict"],
        )

    def test_build_failure_rows_marks_wrong_predictions_and_errors(self) -> None:
        method = MethodSpec("openai_ict", "OpenAI", "openai", "llm_api", "gpt")
        failure_rows = build_failure_rows(
            dataset_name="MetaTool",
            method=method,
            results=[
                {
                    "provider": "openai",
                    "model_name": "gpt",
                    "example_id": "ok",
                    "query": "q0",
                    "expected_tool": "tool_a",
                    "status": "ok",
                    "selected_tool": "tool_a",
                    "ranked_tools": ["tool_a"],
                    "correct_top1": True,
                },
                {
                    "provider": "openai",
                    "model_name": "gpt",
                    "example_id": "wrong",
                    "query": "q1",
                    "expected_tool": "tool_a",
                    "status": "ok",
                    "selected_tool": "tool_b",
                    "ranked_tools": ["tool_b", "tool_a"],
                    "correct_top1": False,
                    "reason": "picked another tool",
                    "latency_ms": 12.5,
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "total_tokens": 14,
                },
                {
                    "provider": "openai",
                    "model_name": "gpt",
                    "example_id": "error",
                    "query": "q2",
                    "expected_tool": "tool_a",
                    "status": "error",
                    "selected_tool": None,
                    "ranked_tools": [],
                    "correct_top1": None,
                    "error_message": "HTTP 429",
                },
            ],
        )

        self.assertEqual([row["example_id"] for row in failure_rows], ["wrong", "error"])
        self.assertEqual(failure_rows[0]["failure_type"], "incorrect_top1")
        self.assertEqual(failure_rows[0]["selected_tool"], "tool_b")
        self.assertEqual(failure_rows[1]["failure_type"], "error")
        self.assertEqual(failure_rows[1]["error_message"], "HTTP 429")

    def test_failure_matrix_includes_only_tasks_with_at_least_one_failure(self) -> None:
        methods = [
            MethodSpec("openai_ict", "OpenAI", "openai", "llm_api", "gpt"),
            MethodSpec("gemini_ict", "Gemini", "gemini", "llm_api", "gemini"),
        ]
        ok_openai = {
            "example_id": "ok",
            "query": "q0",
            "expected_tool": "tool_a",
            "status": "ok",
            "selected_tool": "tool_a",
            "ranked_tools": ["tool_a"],
            "correct_top1": True,
        }
        failed_task_openai = {
            "example_id": "bad",
            "query": "q1",
            "expected_tool": "tool_a",
            "status": "ok",
            "selected_tool": "tool_b",
            "ranked_tools": ["tool_b", "tool_a"],
            "correct_top1": False,
        }
        ok_gemini = {
            "example_id": "ok",
            "query": "q0",
            "expected_tool": "tool_a",
            "status": "ok",
            "selected_tool": "tool_a",
            "ranked_tools": ["tool_a"],
            "correct_top1": True,
        }
        error_gemini = {
            "example_id": "bad",
            "query": "q1",
            "expected_tool": "tool_a",
            "status": "error",
            "selected_tool": None,
            "ranked_tools": [],
            "correct_top1": None,
            "error_message": "HTTP 404",
        }

        matrix_rows = build_failure_matrix_rows(
            dataset_name="MetaTool",
            methods=methods,
            results_by_method={
                "openai_ict": [ok_openai, failed_task_openai],
                "gemini_ict": [ok_gemini, error_gemini],
            },
        )

        self.assertEqual(len(matrix_rows), 1)
        self.assertEqual(matrix_rows[0]["example_id"], "bad")
        self.assertEqual(matrix_rows[0]["openai_ict"], "wrong:tool_b")
        self.assertEqual(matrix_rows[0]["gemini_ict"], "error:HTTP 404")

    def test_latex_table_supports_all_five_methods(self) -> None:
        methods = [
            MethodSpec("qwen_ict", "Qwen3-27B (ICT)", "huggingface", "llm_local", "qwen"),
            MethodSpec("openai_ict", "ChatGPT 5 (ICT)", "openai", "llm_api", "gpt"),
            MethodSpec("gemini_ict", "Gemini 2.5 Flash (ICT)", "gemini", "llm_api", "gemini"),
            MethodSpec("anthropic_ict", "Claude Sonnet 4.6 (ICT)", "anthropic", "llm_api", "claude"),
            MethodSpec("ntilc", "NTILC", "embedding", "embedding", "ntilc"),
        ]
        rows = [
            {
                "dataset": "MetaTool",
                "method_key": method.key,
                "method": method.label,
                "registry_tokens": 0 if method.key == "ntilc" else 100,
                "total_tokens": 10,
                "top_1_accuracy": 1.0,
                "top_5_accuracy": 1.0,
                "latency_ms": 5,
            }
            for method in methods
        ]

        table = build_latex_table(
            rows,
            dataset_order=["MetaTool"],
            method_order=[method.key for method in methods],
        )

        for method in methods:
            self.assertIn(method.label, table)
        self.assertIn(r"\textbf{NTILC}", table)
        self.assertEqual(table.count(r"\\"), 6)


if __name__ == "__main__":
    unittest.main()

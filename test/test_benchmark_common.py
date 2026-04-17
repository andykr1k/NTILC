from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmark.common import (
    build_benchmark_summary,
    discover_embedding_variants,
    parse_selection_response,
    summarize_result_rows,
)


class BenchmarkCommonTests(unittest.TestCase):
    def test_parse_selection_response_normalizes_ranked_tools(self) -> None:
        payload = parse_selection_response(
            '{"selected_tool":"weather","ranked_tools":["weather","weather","stocks","invalid"],"reason":"best fit"}',
            valid_tool_names=["weather", "stocks", "calendar"],
            ranking_limit=3,
        )

        self.assertEqual(payload["selected_tool"], "weather")
        self.assertEqual(payload["ranked_tools"], ["weather", "stocks"])
        self.assertEqual(payload["reason"], "best fit")

    def test_summarize_result_rows_aggregates_metrics(self) -> None:
        rows = [
            {
                "status": "ok",
                "expected_tool": "weather",
                "correct_top1": True,
                "top_3_hit": True,
                "top_5_hit": True,
                "reciprocal_rank": 1.0,
                "latency_ms": 12.0,
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "cost_usd": 0.001,
            },
            {
                "status": "ok",
                "expected_tool": "stocks",
                "correct_top1": False,
                "top_3_hit": True,
                "top_5_hit": True,
                "reciprocal_rank": 0.5,
                "latency_ms": 18.0,
                "input_tokens": 11,
                "output_tokens": 3,
                "total_tokens": 14,
                "cost_usd": 0.002,
            },
            {
                "status": "error",
                "expected_tool": "calendar",
            },
        ]

        metrics = summarize_result_rows(rows)

        self.assertEqual(metrics["total_examples"], 3)
        self.assertEqual(metrics["successful_examples"], 2)
        self.assertEqual(metrics["error_examples"], 1)
        self.assertEqual(metrics["top_1_accuracy"], 0.5)
        self.assertEqual(metrics["top_3_accuracy"], 1.0)
        self.assertEqual(metrics["mean_reciprocal_rank"], 0.75)
        self.assertEqual(metrics["mean_latency_ms"], 15.0)
        self.assertEqual(metrics["sum_total_tokens"], 26.0)
        self.assertEqual(metrics["sum_cost_usd"], 0.003)
        self.assertEqual(metrics["per_tool_accuracy"]["weather"], 1.0)
        self.assertEqual(metrics["per_tool_accuracy"]["stocks"], 0.0)

    def test_discover_embedding_variants_supports_multiple_root_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "output"
            normal_ckpt = output_root / "normal" / "prototype_ce" / "best.pt"
            hier_ckpt = output_root / "hierarchical" / "circle" / "best.pt"
            normal_ckpt.parent.mkdir(parents=True, exist_ok=True)
            hier_ckpt.parent.mkdir(parents=True, exist_ok=True)
            normal_ckpt.write_bytes(b"checkpoint")
            hier_ckpt.write_bytes(b"checkpoint")

            from_output_root = discover_embedding_variants(output_root, "best.pt")
            from_arch_root = discover_embedding_variants(output_root / "normal", "best.pt")
            from_variant_root = discover_embedding_variants(output_root / "normal" / "prototype_ce", "best.pt")

        self.assertEqual([item.variant_id for item in from_output_root], ["hierarchical/circle", "normal/prototype_ce"])
        self.assertEqual([item.variant_id for item in from_arch_root], ["normal/prototype_ce"])
        self.assertEqual([item.variant_id for item in from_variant_root], ["normal/prototype_ce"])

    def test_build_benchmark_summary_includes_leaderboard(self) -> None:
        summary = build_benchmark_summary(
            benchmark_name="tool_selection",
            dataset_path=Path("dataset.jsonl"),
            tools_path=Path("tools.json"),
            output_dir=Path("benchmark/output/run"),
            config={"ranking_limit": 5},
            model_summaries=[
                {
                    "adapter_id": "embedding/a",
                    "provider": "embedding",
                    "mode": "embedding",
                    "model_name": "a",
                    "status": "ok",
                    "metrics": {
                        "top_1_accuracy": 0.7,
                        "top_3_accuracy": 0.9,
                        "top_5_accuracy": 1.0,
                        "mean_reciprocal_rank": 0.8,
                        "mean_latency_ms": 10.0,
                        "mean_total_tokens": 12.0,
                        "sum_cost_usd": None,
                    },
                },
                {
                    "adapter_id": "embedding/b",
                    "provider": "embedding",
                    "mode": "embedding",
                    "model_name": "b",
                    "status": "ok",
                    "metrics": {
                        "top_1_accuracy": 0.5,
                        "top_3_accuracy": 0.8,
                        "top_5_accuracy": 0.9,
                        "mean_reciprocal_rank": 0.6,
                        "mean_latency_ms": 8.0,
                        "mean_total_tokens": 12.0,
                        "sum_cost_usd": None,
                    },
                },
            ],
        )

        self.assertEqual(summary["counts"]["model_count"], 2)
        self.assertEqual(summary["leaderboard"][0]["adapter_id"], "embedding/a")
        self.assertEqual(summary["leaderboard"][1]["adapter_id"], "embedding/b")


if __name__ == "__main__":
    unittest.main()

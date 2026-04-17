from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.dataset_utils import clean_rows, load_dataset_rows, split_rows_by_tool


class TestDatasetUtils(unittest.TestCase):
    def test_load_dataset_rows_supports_json_and_jsonl(self) -> None:
        rows = [
            {"tool": "alpha", "query": "alpha query"},
            {"tool": "beta", "query": "beta query"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            jsonl_path = root / "dataset.jsonl"
            json_path = root / "dataset.json"

            with jsonl_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row))
                    handle.write("\n")

            json_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

            self.assertEqual(load_dataset_rows(jsonl_path), rows)
            self.assertEqual(load_dataset_rows(json_path), rows)

    def test_clean_rows_normalizes_query_field(self) -> None:
        rows = [
            {"tool": "alpha", "text": "alpha query"},
            {"tool": "beta", "query": "beta query"},
            {"tool": "", "query": "skip me"},
        ]

        cleaned = clean_rows(rows)

        self.assertEqual(
            cleaned,
            [
                {"tool": "alpha", "text": "alpha query", "query": "alpha query"},
                {"tool": "beta", "query": "beta query"},
            ],
        )

    def test_split_rows_by_tool_creates_fixed_balanced_split(self) -> None:
        rows = [
            {"tool": tool_name, "query": f"{tool_name} query {index}"}
            for tool_name in ("alpha", "beta")
            for index in range(20)
        ]

        train_rows, test_rows, summary = split_rows_by_tool(
            rows,
            examples_per_tool=20,
            test_per_tool=4,
            seed=7,
        )

        self.assertEqual(len(train_rows), 32)
        self.assertEqual(len(test_rows), 8)
        self.assertEqual(summary["train_per_tool"], 16)
        self.assertEqual(summary["test_per_tool"], 4)
        self.assertEqual(summary["train_counts_per_tool"], {"alpha": 16, "beta": 16})
        self.assertEqual(summary["test_counts_per_tool"], {"alpha": 4, "beta": 4})
        self.assertTrue(all(row["split"] == "train" for row in train_rows))
        self.assertTrue(all(row["split"] == "test" for row in test_rows))

    def test_split_rows_by_tool_requires_enough_examples_per_tool(self) -> None:
        rows = [
            {"tool": "alpha", "query": f"alpha query {index}"}
            for index in range(12)
        ]

        with self.assertRaisesRegex(ValueError, "has only 12 rows"):
            split_rows_by_tool(
                rows,
                examples_per_tool=20,
                test_per_tool=4,
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()

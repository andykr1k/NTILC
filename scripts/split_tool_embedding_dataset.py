#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dataset_utils import (
    clean_rows,
    load_dataset_rows,
    split_rows_by_tool,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create explicit train/test splits for a tool embedding dataset by selecting a fixed "
            "number of queries per tool and holding out a fixed number for test."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the source dataset (.json or .jsonl).",
    )
    parser.add_argument(
        "--train-output-path",
        type=Path,
        default=None,
        help="Where to write the train split JSONL. Defaults to <dataset>_train.jsonl.",
    )
    parser.add_argument(
        "--test-output-path",
        type=Path,
        default=None,
        help="Where to write the test split JSONL. Defaults to <dataset>_test.jsonl.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Where to write the split summary JSON. Defaults to <dataset>_split_summary.json.",
    )
    parser.add_argument(
        "--examples-per-tool",
        type=int,
        default=20,
        help="How many total examples to keep per tool before splitting.",
    )
    parser.add_argument(
        "--test-per-tool",
        type=int,
        default=4,
        help="How many examples per tool to hold out for the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for per-tool sampling and shuffling.",
    )
    return parser.parse_args()


def default_output_path(dataset_path: Path, suffix: str) -> Path:
    if dataset_path.suffix:
        return dataset_path.with_name(f"{dataset_path.stem}{suffix}")
    return dataset_path.with_name(f"{dataset_path.name}{suffix}")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.resolve()
    train_output_path = (
        args.train_output_path.resolve()
        if args.train_output_path is not None
        else default_output_path(dataset_path, "_train.jsonl")
    )
    test_output_path = (
        args.test_output_path.resolve()
        if args.test_output_path is not None
        else default_output_path(dataset_path, "_test.jsonl")
    )
    summary_path = (
        args.summary_path.resolve()
        if args.summary_path is not None
        else default_output_path(dataset_path, "_split_summary.json")
    )

    source_rows = clean_rows(load_dataset_rows(dataset_path))
    train_rows, test_rows, summary = split_rows_by_tool(
        source_rows,
        examples_per_tool=args.examples_per_tool,
        test_per_tool=args.test_per_tool,
        seed=args.seed,
    )

    summary.update(
        {
            "dataset_path": str(dataset_path),
            "train_output_path": str(train_output_path),
            "test_output_path": str(test_output_path),
            "summary_path": str(summary_path),
        }
    )

    write_jsonl(train_output_path, train_rows)
    write_jsonl(test_output_path, test_rows)
    write_json(summary_path, summary)

    print(f"Wrote train split to {train_output_path}")
    print(f"Wrote test split to {test_output_path}")
    print(f"Wrote split summary to {summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - depends on local test environment
    torch = None
    F = None

if torch is not None:
    from training.train_embedding_space import (
        clean_rows,
        compute_loss,
        load_jsonl,
        prepare_tool_compatibility_matrix,
    )
else:  # pragma: no cover - depends on local test environment
    clean_rows = None
    compute_loss = None
    load_jsonl = None
    prepare_tool_compatibility_matrix = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATASET_PATH = PROJECT_ROOT / "data" / "OSS" / "tool_embedding_dataset.jsonl"
REAL_TOOLS_PATH = PROJECT_ROOT / "data" / "OSS" / "tools.json"
REAL_METRICS_PATH = PROJECT_ROOT / "data" / "OSS" / "output" / "normal" / "functional_margin" / "metrics.json"


class DummyPrototypeModel:
    def __init__(self, prototypes: torch.Tensor) -> None:
        self.tool_prototypes = prototypes


@unittest.skipUnless(REAL_METRICS_PATH.exists(), "real functional_margin metrics were not found under data/OSS")
class TestFunctionalMarginLossRunArtifacts(unittest.TestCase):
    def test_real_training_metrics_show_inactive_functional_term(self) -> None:
        history = json.loads(REAL_METRICS_PATH.read_text(encoding="utf-8"))
        self.assertTrue(history)

        penalties = [float(row.get("train_functional_penalty", 0.0) or 0.0) for row in history]
        active_margins = [float(row.get("train_active_functional_margin", 0.0) or 0.0) for row in history]
        incompatibilities = [
            float(row.get("train_mean_negative_incompatibility", 0.0) or 0.0)
            for row in history
        ]

        self.assertTrue(all(abs(value) < 1e-12 for value in penalties))
        self.assertTrue(all(abs(value) < 1e-12 for value in active_margins))
        self.assertTrue(all(value > 0.0 for value in incompatibilities))


class TestFunctionalMarginLoss(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    @unittest.skipUnless(
        REAL_DATASET_PATH.exists() and REAL_TOOLS_PATH.exists(),
        "real OSS dataset/tools.json were not found under data/OSS",
    )
    def test_real_data_builds_nontrivial_incompatibility_weights(self) -> None:
        rows = clean_rows(load_jsonl(REAL_DATASET_PATH))
        tool_names = sorted({row["tool"] for row in rows})

        matrix, info = prepare_tool_compatibility_matrix(
            tool_names=tool_names,
            rows=rows,
            dataset_path=REAL_DATASET_PATH,
            tools_path=str(REAL_TOOLS_PATH),
        )

        off_diagonal = matrix[~torch.eye(matrix.size(0), dtype=torch.bool)]
        incompatibility = 1.0 - off_diagonal

        self.assertEqual(info["metadata_source"], "tools.json")
        self.assertTrue(torch.allclose(torch.diag(matrix), torch.ones(matrix.size(0))))
        self.assertGreater(float(incompatibility.mean()), 0.0)
        self.assertLess(float(off_diagonal.max()), 1.0)

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_prepare_tool_compatibility_matrix_builds_nontrivial_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_path = root / "tool_embedding_dataset.jsonl"
            tools_path = root / "tools.json"
            dataset_path.write_text("", encoding="utf-8")
            tools_path.write_text(
                json.dumps(
                    {
                        "tools": [
                            {
                                "id": "delete_file",
                                "interface_type": "python",
                                "parent_id": "file_system",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"},
                                        "recursive": {"type": "boolean"},
                                    },
                                    "required": ["path"],
                                },
                            },
                            {
                                "id": "fetch_url",
                                "interface_type": "http",
                                "parent_id": "information_retrieval",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "url": {"type": "string"},
                                        "format": {
                                            "type": "string",
                                            "enum": ["text", "markdown", "html"],
                                        },
                                    },
                                    "required": ["url"],
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = [
                {
                    "tool": "delete_file",
                    "query": "Delete report.txt from disk.",
                    "interface_type": "python",
                    "parent_id": "file_system",
                },
                {
                    "tool": "fetch_url",
                    "query": "Fetch https://example.com as markdown.",
                    "interface_type": "http",
                    "parent_id": "information_retrieval",
                },
            ]

            matrix, info = prepare_tool_compatibility_matrix(
                tool_names=["delete_file", "fetch_url"],
                rows=rows,
                dataset_path=dataset_path,
                tools_path=str(tools_path),
            )

        self.assertEqual(info["metadata_source"], "tools.json")
        self.assertTrue(torch.allclose(torch.diag(matrix), torch.ones(2)))
        self.assertGreaterEqual(float(matrix.min()), 0.0)
        self.assertLessEqual(float(matrix.max()), 1.0)
        self.assertLess(float(matrix[0, 1]), 0.2)

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_functional_margin_penalty_stays_zero_when_negatives_are_outside_margin(self) -> None:
        embeddings = F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), dim=-1)
        prototypes = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.30, 0.9539392],
                    [-1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            dim=-1,
        )
        model = DummyPrototypeModel(prototypes)
        logits = embeddings @ F.normalize(model.tool_prototypes, dim=-1).T
        compatibility_matrix = torch.tensor(
            [
                [1.0, 0.05, 0.05],
                [0.05, 1.0, 0.20],
                [0.05, 0.20, 1.0],
            ],
            dtype=torch.float32,
        )

        loss, metrics = compute_loss(
            model=model,
            embeddings=embeddings,
            logits=logits,
            labels=torch.tensor([0], dtype=torch.long),
            loss_type="functional_margin",
            alignment_weight=0.0,
            contrastive_margin=0.5,
            circle_margin=0.25,
            circle_gamma=32.0,
            compatibility_matrix=compatibility_matrix,
            compatibility_weight=1.0,
            compatibility_margin=0.5,
        )

        self.assertGreater(metrics["mean_negative_incompatibility"], 0.0)
        self.assertAlmostEqual(metrics["functional_penalty"], 0.0, places=7)
        self.assertAlmostEqual(metrics["active_functional_margin"], 0.0, places=7)
        self.assertAlmostEqual(float(loss), metrics["semantic_cross_entropy"], places=6)

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_functional_margin_penalty_turns_on_when_risky_negative_enters_margin(self) -> None:
        embeddings = F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), dim=-1)
        prototypes = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.80, 0.60],
                    [-1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            dim=-1,
        )
        model = DummyPrototypeModel(prototypes)
        logits = embeddings @ F.normalize(model.tool_prototypes, dim=-1).T
        compatibility_matrix = torch.tensor(
            [
                [1.0, 0.10, 0.10],
                [0.10, 1.0, 0.20],
                [0.10, 0.20, 1.0],
            ],
            dtype=torch.float32,
        )

        loss, metrics = compute_loss(
            model=model,
            embeddings=embeddings,
            logits=logits,
            labels=torch.tensor([0], dtype=torch.long),
            loss_type="functional_margin",
            alignment_weight=0.0,
            contrastive_margin=0.5,
            circle_margin=0.25,
            circle_gamma=32.0,
            compatibility_matrix=compatibility_matrix,
            compatibility_weight=1.0,
            compatibility_margin=0.5,
        )

        self.assertGreater(metrics["mean_negative_incompatibility"], 0.0)
        self.assertGreater(metrics["functional_penalty"], 0.0)
        self.assertGreater(metrics["active_functional_margin"], 0.0)
        self.assertGreater(float(loss), metrics["semantic_cross_entropy"])

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_functional_margin_normalizes_by_active_negative_weights_only(self) -> None:
        embeddings = F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), dim=-1)
        prototypes = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.80, 0.60],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            dim=-1,
        )
        model = DummyPrototypeModel(prototypes)
        logits = embeddings @ F.normalize(model.tool_prototypes, dim=-1).T
        compatibility_matrix = torch.tensor(
            [
                [1.0, 0.10, 0.10],
                [0.10, 1.0, 0.20],
                [0.10, 0.20, 1.0],
            ],
            dtype=torch.float32,
        )

        loss, metrics = compute_loss(
            model=model,
            embeddings=embeddings,
            logits=logits,
            labels=torch.tensor([0], dtype=torch.long),
            loss_type="functional_margin",
            alignment_weight=0.0,
            contrastive_margin=0.5,
            circle_margin=0.25,
            circle_gamma=32.0,
            compatibility_matrix=compatibility_matrix,
            compatibility_weight=1.0,
            compatibility_margin=0.5,
        )

        self.assertAlmostEqual(metrics["active_functional_margin"], 0.3, places=6)
        self.assertAlmostEqual(metrics["functional_penalty"], 0.09, places=6)
        self.assertGreater(float(loss), metrics["semantic_cross_entropy"])


if __name__ == "__main__":
    unittest.main()

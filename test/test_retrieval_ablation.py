from __future__ import annotations

import unittest

from benchmark.retrieval_ablation import (
    BM25Retriever,
    build_semantic_blur_index,
    is_signature_error,
    signature_atoms,
    signatures_compatible,
    summarize_ablation_results,
)


def tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


class RetrievalAblationTests(unittest.TestCase):
    def test_signature_atoms_capture_required_type_contract(self) -> None:
        weather_zip = tool(
            "weather_zip",
            "Get weather",
            {"zip_code": {"type": "string"}},
            ["zip_code"],
        )
        weather_latlon = tool(
            "weather_latlon",
            "Get weather",
            {"lat": {"type": "number"}, "lon": {"type": "number"}},
            ["lat", "lon"],
        )
        weather_zip_copy = tool(
            "weather_zip_copy",
            "Get weather by zip",
            {"zip_code": {"type": "string"}},
            ["zip_code"],
        )

        self.assertIn("required:zip_code:string:", signature_atoms(weather_zip))
        self.assertTrue(signatures_compatible(weather_zip, weather_zip_copy))
        self.assertFalse(signatures_compatible(weather_zip, weather_latlon))

    def test_signature_error_ignores_wrong_tool_with_same_contract(self) -> None:
        search_a = tool("search_a", "Search A", {"query": {"type": "string"}}, ["query"])
        search_b = tool("search_b", "Search B", {"query": {"type": "string"}}, ["query"])
        search_by_id = tool("search_by_id", "Search by ID", {"id": {"type": "integer"}}, ["id"])
        lookup = {item["name"]: item for item in (search_a, search_b, search_by_id)}

        self.assertFalse(is_signature_error(lookup, "search_a", "search_b"))
        self.assertTrue(is_signature_error(lookup, "search_a", "search_by_id"))

    def test_bm25_ranks_schema_text(self) -> None:
        retriever = BM25Retriever(
            ["weather", "calendar"],
            ["weather forecast zipcode storm", "calendar event invite date"],
        )

        ranked = retriever.rank("storm forecast", top_k=2)

        self.assertEqual(ranked[0][0], "weather")

    def test_semantic_blur_index_keeps_similar_incompatible_tools(self) -> None:
        tools = [
            tool("weather_zip", "Get weather forecast", {"zip_code": {"type": "string"}}, ["zip_code"]),
            tool(
                "weather_latlon",
                "Get weather forecast",
                {"lat": {"type": "number"}, "lon": {"type": "number"}},
                ["lat", "lon"],
            ),
            tool("calculator", "Evaluate math expression", {"expression": {"type": "string"}}, ["expression"]),
        ]

        blur_index = build_semantic_blur_index(
            tools,
            similarity_threshold=0.1,
            top_n=3,
        )

        self.assertIn("weather_zip", blur_index["blur_tools"])
        self.assertEqual(blur_index["blur_tools"]["weather_zip"][0]["tool"], "weather_latlon")

    def test_summarize_ablation_results_adds_table_metrics(self) -> None:
        metrics = summarize_ablation_results(
            [
                {
                    "status": "ok",
                    "correct_top1": True,
                    "top_5_hit": True,
                    "semantic_blur_case": True,
                    "semantic_blur_hit": True,
                    "signature_error": False,
                    "latency_ms": 10.0,
                },
                {
                    "status": "ok",
                    "correct_top1": False,
                    "top_5_hit": True,
                    "semantic_blur_case": True,
                    "semantic_blur_hit": False,
                    "signature_error": True,
                    "latency_ms": 20.0,
                },
            ]
        )

        self.assertEqual(metrics["top_1_accuracy"], 0.5)
        self.assertEqual(metrics["top_5_accuracy"], 1.0)
        self.assertEqual(metrics["semantic_blur_accuracy"], 0.5)
        self.assertEqual(metrics["signature_error_rate"], 0.5)
        self.assertEqual(metrics["mean_latency_ms"], 15.0)


if __name__ == "__main__":
    unittest.main()

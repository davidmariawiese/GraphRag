"""Unittest-discoverable smoke checks.

Goals:
- Ensure core modules import.
- Ensure `build_pipeline()` can construct the Haystack pipeline.

This file intentionally avoids network calls to Neo4j/Ollama; it only constructs
objects and inspects the pipeline graph.
"""

from __future__ import annotations

import importlib
import os
import unittest


class TestImportsAndWiring(unittest.TestCase):
    def test_imports(self) -> None:
        modules = [
            "src.config",
            "src.base.base_pipeline",
            "src.graph.query_processor",
            "src.graph.retriever",
            "src.graph.organizer",
            "src.graph.generator",
            "src.graph.graph_pipeline",
        ]

        for name in modules:
            importlib.import_module(name)

    def test_build_pipeline_wiring(self) -> None:
        # Use localhost defaults to avoid relying on container DNS names.
        # Construction should not trigger network calls.
        os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
        os.environ.setdefault("NEO4J_USER", "neo4j")
        os.environ.setdefault("NEO4J_PASSWORD", "password")
        os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
        os.environ.setdefault("OLLAMA_QUERY_MODEL", "llama3.1:8b")
        os.environ.setdefault("OLLAMA_GENERATOR_MODEL", "llama3.1:8b")

        from src.graph.graph_pipeline import build_pipeline

        pipe = build_pipeline()
        self.assertTrue(hasattr(pipe, "run"))
        self.assertTrue(hasattr(pipe, "graph"))

        # Haystack Pipeline exposes a networkx graph with component names.
        nodes = set(pipe.graph.nodes)
        self.assertEqual(nodes, {"query_processor", "retriever", "organizer", "generator"})

        edges = {(u, v, key) for (u, v, key) in pipe.graph.edges(keys=True)}
        expected_edges = {
            ("query_processor", "retriever", "cypher/cypher"),
            ("retriever", "organizer", "documents/documents"),
            ("organizer", "generator", "analysis_json/analysis_json"),
            ("query_processor", "generator", "cypher/cypher_query"),
        }
        self.assertEqual(edges, expected_edges)


if __name__ == "__main__":
    unittest.main()

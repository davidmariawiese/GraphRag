"""Minimal smoke test for importability.

Run:
  python -m tests.test_imports

This avoids network calls to Neo4j/Ollama; it only checks that modules import
and the pipeline can be constructed.
"""

from __future__ import annotations


def main() -> None:
    import importlib

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

    from src.graph.graph_pipeline import build_pipeline

    pipe = build_pipeline()
    assert pipe is not None
    assert hasattr(pipe, "run")

    print("OK: imports + build_pipeline()")


if __name__ == "__main__":
    main()

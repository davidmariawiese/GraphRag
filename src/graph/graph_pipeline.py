"""GraphRAG pipeline wiring.

This module defines the Haystack pipeline composition for GraphRAG:

1) `QueryProcessor` turns a user question into a (read-only) Cypher query.
2) `CypherRetriever` executes the Cypher query against Neo4j and returns documents.
3) `Organizer` deterministically aggregates/structures the result into JSON.
4) `Generator` renders a German answer based on the JSON context.

The pipeline is primarily configured via environment variables (see `src/config.py`).
"""

import os
from typing import Optional

from dotenv import load_dotenv
from haystack import Pipeline

from ..config import (
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USER,
    GRAPH_RAG_DEBUG_ENV,
    NEO4J_PASSWORD_ENV,
    NEO4J_URI_ENV,
    NEO4J_USER_ENV,
)
from .generator import Generator
from .organizer import Organizer
from .query_processor import QueryProcessor
from .retriever import CypherRetriever

load_dotenv()


def build_pipeline() -> Pipeline:
    """Build and connect the GraphRAG Haystack pipeline."""
    pipeline = Pipeline()

    pipeline.add_component("query_processor", QueryProcessor())
    pipeline.add_component(
        "retriever",
        CypherRetriever(
            uri=os.getenv(NEO4J_URI_ENV, DEFAULT_NEO4J_URI),
            user=os.getenv(NEO4J_USER_ENV, DEFAULT_NEO4J_USER),
            password=os.getenv(NEO4J_PASSWORD_ENV, DEFAULT_NEO4J_PASSWORD),
        ),
    )
    pipeline.add_component("organizer", Organizer())
    pipeline.add_component("generator", Generator())

    # CONNECT COMPONENTS (Output → Input)
    pipeline.connect("query_processor.cypher", "retriever.cypher")
    pipeline.connect("retriever.documents", "organizer.documents")
    pipeline.connect("organizer.analysis_json", "generator.analysis_json")
    pipeline.connect("query_processor.cypher", "generator.cypher_query")
    return pipeline


def run_pipeline(pipe: Pipeline, query: str, debug: Optional[bool] = None) -> str:
    """Führt die Haystack-2.x-Pipeline aus und gibt die Antwort zurück.

    Wenn `debug=True` (oder via env GRAPH_RAG_DEBUG), geben alle Komponenten live ihre Inputs auf stdout aus.
    """
    env_key = GRAPH_RAG_DEBUG_ENV

    # Determine effective debug state: arg overrides env, env defaults to False
    if debug is None:
        should_debug = os.getenv(env_key) == "true"
    else:
        should_debug = debug

    prev = os.environ.get(env_key)
    try:
        if should_debug:
            os.environ[env_key] = "true"
        else:
            # Ensure it's off for this run if explicitly disabled or default off
            if env_key in os.environ:
                os.environ.pop(env_key)

        result = pipe.run(
            {
                "query_processor": {"query": query},
                "generator": {"query": query},
            }
        )
        return result["generator"]["answer"]
    finally:
        # Restore original environment state
        if prev is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = prev

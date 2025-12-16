from haystack import Pipeline
from .query_processor import QueryProcessor
from .retriever import CypherRetriever
from .organizer import Organizer
from .generator import Generator
import os
from dotenv import load_dotenv

load_dotenv()


def build_pipeline():

    pipeline = Pipeline()

    pipeline.add_component("query_processor", QueryProcessor())
    pipeline.add_component("retriever", CypherRetriever(
        uri=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password")
    ))
    pipeline.add_component("organizer", Organizer())
    pipeline.add_component("generator", Generator())

    # CONNECT COMPONENTS (Output → Input)
    pipeline.connect("query_processor.cypher", "retriever.cypher")
    pipeline.connect("retriever.documents", "organizer.documents")
    pipeline.connect("organizer.analysis_json", "generator.analysis_json")
    pipeline.connect("query_processor.cypher", "generator.cypher_query") 
    return pipeline

def run_pipeline(pipe, query: str, debug: bool | None = None):
    """Führt die Haystack-2.x-Pipeline aus und gibt die Antwort zurück.

    Wenn `debug=True` (oder via env GRAPH_RAG_DEBUG), geben alle Komponenten live ihre Inputs auf stdout aus.
    """
    env_key = "GRAPH_RAG_DEBUG"
    
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

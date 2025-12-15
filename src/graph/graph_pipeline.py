from haystack import Pipeline
from .query_processor import QueryProcessor
from .retriever import CypherRetriever
from .organizer import Organizer
from .generator import Generator
import os


def build_pipeline():

    pipeline = Pipeline()

    pipeline.add_component("query_processor", QueryProcessor())
    pipeline.add_component("retriever", CypherRetriever(
        uri="bolt://neo4j:7687",
        user="neo4j",
        password="password"
    ))
    pipeline.add_component("organizer", Organizer())
    pipeline.add_component("generator", Generator())

    # CONNECT COMPONENTS (Output → Input)
    pipeline.connect("query_processor.cypher", "retriever.cypher")
    pipeline.connect("retriever.documents", "organizer.documents")
    pipeline.connect("organizer.context", "generator.context")
    pipeline.connect("query_processor.cypher", "generator.cypher_query") 
    return pipeline

def run_pipeline(pipe, query: str, debug: bool = False):
    """Führt die Haystack-2.x-Pipeline aus und gibt die Antwort zurück.

    Wenn `debug=True`, geben alle Komponenten live ihre Inputs auf stdout aus.
    """
    env_key = "GRAPH_RAG_DEBUG"
    prev = os.environ.get(env_key)
    try:
        if debug:
            os.environ[env_key] = "1"
        else:
            os.environ.pop(env_key, None)

        result = pipe.run(
            {
                "query_processor": {"query": query},
                "generator": {"query": query},
            }
        )
        return result["generator"]["answer"]
    finally:
        if prev is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = prev

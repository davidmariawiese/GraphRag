from haystack import Pipeline
from .query_processor import QueryProcessor
from .cypher_retriever import CypherRetriever
from .organizer import Organizer
from .generator import Generator


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

def run_pipeline(pipe, query: str):
    """
    Führt die Haystack-2.x-Pipeline aus und gibt die Antwort zurück.
    """
    result = pipe.run(
        {
            "query_processor": {"query": query},
            "generator": {"query": query}
        }
    )

    # Der Generator enthält die finale Antwort
    return result["generator"]["answer"]

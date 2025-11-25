from haystack import Pipeline

from .cypher_retriever import CypherRetriever
from .generator import Generator
from .query_processor import QueryProcessor


def build_pipeline(uri: str = "bolt://neo4j:7687",
                   user: str = "neo4j",
                   password: str = "password") -> Pipeline:
    """Construct the GraphRAG pipeline with Haystack 2.x components."""

    pipeline = Pipeline()

    planner = QueryProcessor()
    retriever = CypherRetriever(uri=uri, user=user, password=password)
    generator = Generator()

    pipeline.add_component("planner", planner)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("generator", generator)

    pipeline.connect("planner.cypher", "retriever.cypher")
    pipeline.connect("retriever.documents", "generator.documents")

def run_pipeline(pipe, query):
  result = pipe.run({"prompt_builder": {"query": query},"retriever": {"query": query}})
  return result
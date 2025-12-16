from haystack import component
from haystack.dataclasses import Document
from neo4j import GraphDatabase
import os


def _debug_enabled() -> bool:
    return os.getenv("GRAPH_RAG_DEBUG") == "true"

@component
class CypherRetriever:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @component.output_types(documents=list)
    def run(self, cypher: str):

        if _debug_enabled():
            print("\n[retriever] input.cypher:")
            print(cypher)

        with self.driver.session() as session:
            result = session.run(cypher).data()
        # print(result)
        docs = [Document(content=str(r)) for r in result]
        return {"documents": docs}

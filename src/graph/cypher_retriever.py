from haystack import component
from haystack.dataclasses import Document
from neo4j import GraphDatabase

@component
class CypherRetriever:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @component.output_types(documents=list)
    def run(self, cypher: str):

        with self.driver.session() as session:
            result = session.run(cypher).data()

        docs = [Document(content=str(r)) for r in result]
        return {"documents": docs}

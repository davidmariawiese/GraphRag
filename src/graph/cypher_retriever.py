from typing import List

from haystack import Document, component
from neo4j import GraphDatabase


@component
class CypherRetriever:
    """
    Einfacher Custom-Component für Haystack 2.x:
    - nimmt einen Cypher-String als Input
    - führt ihn in Neo4j aus
    - gibt eine Liste von Documents zurück
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @component.output_types(documents=List[Document])
    def run(self, cypher: str):
        # cypher ist ein String, den du z.B. vom Query-Planner bekommst
        with self.driver.session() as session:
            result = session.run(cypher)
            records = result.data()

        docs = [Document(content=str(record)) for record in records]
        return {"documents": docs}

from haystack import component
from haystack.dataclasses import Document
from neo4j import GraphDatabase
import os
import json
from typing import List


def _debug_enabled() -> bool:
    return os.getenv("GRAPH_RAG_DEBUG") == "true"

@component
class CypherRetriever:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @component.output_types(documents=List[Document])
    def run(self, cypher: str):

        if _debug_enabled():
            print("\n[retriever] input.cypher:")
            print(cypher)

        with self.driver.session() as session:
            result = session.run(cypher).data()

        # Serialize Neo4j rows as JSON strings to keep structure and ensure organizer can parse
        docs = []
        for row in result:
            try:
                docs.append(Document(content=json.dumps(row)))
            except Exception:
                # Fallback: plain string serialization
                docs.append(Document(content=str(row)))

        if _debug_enabled():
            print("[retriever] output.documents_total:", len(docs))
            if docs:
                try:
                    preview = docs[0].content
                    suffix = "" if len(preview) <= 800 else "..."
                    print("[retriever] output.first_document:")
                    print(preview[:800] + suffix)
                except Exception:
                    print("[retriever] output.first_document: <unprintable>")
        return {"documents": docs}

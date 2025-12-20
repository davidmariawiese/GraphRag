"""Neo4j Cypher retriever for GraphRAG.

This component executes a Cypher query against Neo4j and returns result rows as
Haystack `Document`s. Each document's `content` is a JSON string representing one
Neo4j record.
"""

import json
import os
from typing import List

from haystack import component
from haystack.dataclasses import Document
from neo4j import GraphDatabase

from ..config import GRAPH_RAG_DEBUG_ENV


_DOC_PREVIEW_CHARS = 800


def _debug_enabled() -> bool:
    return os.getenv(GRAPH_RAG_DEBUG_ENV) == "true"


@component
class CypherRetriever:

    """Execute Cypher queries and return documents."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @component.output_types(documents=List[Document])
    def run(self, cypher: str) -> dict[str, List[Document]]:

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
                    suffix = "" if len(preview) <= _DOC_PREVIEW_CHARS else "..."
                    print("[retriever] output.first_document:")
                    print(preview[:_DOC_PREVIEW_CHARS] + suffix)
                except Exception:
                    print("[retriever] output.first_document: <unprintable>")
        return {"documents": docs}

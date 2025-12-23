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
from neo4j.graph import Node, Relationship, Path

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

        # Normalize Neo4j driver rows into plain Python dicts suitable for JSON.
        def _normalize_value(v):
            # Recursively normalize node/relationship/path/list/dict values.
            if v is None:
                return None
            # Node
            if Node is not None and isinstance(v, Node):
                try:
                    props = dict(v)
                except Exception:
                    props = {k: getattr(v, k, None) for k in dir(v) if not k.startswith("_")}
                try:
                    props["_labels"] = list(v.labels)
                except Exception:
                    pass
                try:
                    props["_id"] = getattr(v, "id", None)
                except Exception:
                    pass
                return props
            # Relationship
            if Relationship is not None and isinstance(v, Relationship):
                try:
                    props = dict(v)
                except Exception:
                    props = {}
                try:
                    start = v.start_node
                    end = v.end_node
                    props["_start"] = dict(start) if start is not None else None
                    props["_end"] = dict(end) if end is not None else None
                except Exception:
                    pass
                return props
            # Path
            if Path is not None and isinstance(v, Path):
                try:
                    nodes = [dict(n) for n in v.nodes]
                    return {"_path_nodes": nodes}
                except Exception:
                    return str(v)
            # Lists
            if isinstance(v, list):
                return [_normalize_value(x) for x in v]
            # Dicts: normalize values
            if isinstance(v, dict):
                return {kk: _normalize_value(vv) for kk, vv in v.items()}
            # Primitive types: return as-is
            return v

        # Apply normalization to each result row
        normalized_rows = []
        for row in result:
            nr = {}
            for k, v in row.items():
                nr[k] = _normalize_value(v)
            normalized_rows.append(nr)

        # Post-normalize rows to a minimal, pipeline-wide canonical shape:
        # - dotted keys (e.g. "m.temperature") -> nested dicts under "Measurement"/"Sensor"
        # - flat measurement fields (time, temperature, humidity, light, voltage)
        #   are collected into `Measurement` dict
        # - alias rename: `m` -> `Measurement`, `s` -> `Sensor`
        # - wrapper rows with `records` are expanded into individual rows
        alias_map = {"m": "Measurement", "Measurement": "Measurement", "s": "Sensor", "Sensor": "Sensor"}

        expanded_rows = []
        for row in normalized_rows:
            # 1) dotted keys -> nested
            for key in list(row.keys()):
                if "." in key:
                    prefix, suffix = key.split(".", 1)
                    mapped_prefix = alias_map.get(prefix, prefix)
                    container = row.setdefault(mapped_prefix, {})
                    if suffix not in container:
                        container[suffix] = row.pop(key)

            # 2) gather flat measurement fields into Measurement
            meas = row.get("Measurement")
            if meas is None:
                meas = {}
            # move known measurement keys
            for field in ("time", "temperature", "humidity", "light", "voltage"):
                if field in row and field not in meas:
                    meas[field] = row.pop(field)
            if meas:
                row["Measurement"] = meas

            # 3) rename short aliases
            if "m" in row and isinstance(row.get("m"), dict) and "Measurement" not in row:
                row["Measurement"] = row.pop("m")
            if "s" in row and isinstance(row.get("s"), dict) and "Sensor" not in row:
                row["Sensor"] = row.pop("s")

            # 4) expand wrapper rows
            if "records" in row and isinstance(row.get("records"), list):
                for rec in row["records"]:
                    expanded_rows.append(rec if isinstance(rec, dict) else {"value": rec})
                continue

            expanded_rows.append(row)

        # Serialize expanded rows to Documents.
        docs = []
        skipped = 0
        for row in expanded_rows:
            try:
                docs.append(Document(content=json.dumps(row)))
            except Exception:
                skipped += 1
                if _debug_enabled():
                    try:
                        print("[retriever] skipped_unserializable_row:", repr(row)[:1000])
                    except Exception:
                        print("[retriever] skipped_unserializable_row: <unprintable>")
                continue
        if _debug_enabled() and skipped:
            print(f"[retriever] skipped_unserializable_rows: {skipped}")

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

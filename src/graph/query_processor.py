"""GraphRAG query-to-Cypher component.

This component uses an LLM (Ollama via Haystack integration) to generate exactly
one Cypher query from a natural-language question.
"""

from haystack import component
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders import PromptBuilder
import os
import re

from ..config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    GRAPH_RAG_DEBUG_ENV,
    OLLAMA_MODEL_ENV,
    OLLAMA_QUERY_MODEL_ENV,
    OLLAMA_URL_ENV,
)


def _debug_enabled() -> bool:
    return os.getenv(GRAPH_RAG_DEBUG_ENV) == "true"


_DEFAULT_QUERY_NUM_PREDICT = 256
_DEFAULT_QUERY_TEMPERATURE = 0.0


_NUMERIC_LITERAL_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _unquote_numeric_literals_in_cypher(cypher: str) -> str:
    """Undo a common LLM mistake: quoting numeric literals (e.g. mote_id: "1.0").

    Neo4j property comparisons are type-sensitive for strings vs numbers.
    The dataset uses numeric IDs (e.g. 1, 7, 7.0), so we normalize numeric
    strings to numeric literals for known numeric keys.
    """
    if not cypher:
        return ""

    # Map patterns: {mote_id: "1.0"}
    cypher = re.sub(
        r'(\b(?:mote_id|moteid|epoch_id|week|hour|minute)\b\s*:\s*)"(-?\d+(?:\.\d+)?)"',
        r"\1\2",
        cypher,
        flags=re.IGNORECASE,
    )

    # IN lists: WHERE s.mote_id IN ["1.0", "2.0"]
    in_list_re = re.compile(
        r'(\b(?:s|m)\.(?:mote_id|moteid)\s+IN\s*\[)([^\]]*)(\])',
        flags=re.IGNORECASE,
    )

    def _fix_in_list(match: re.Match[str]) -> str:
        prefix, items, suffix = match.group(1), match.group(2), match.group(3)
        items = re.sub(r'"(-?\d+(?:\.\d+)?)"', r"\1", items)
        return prefix + items + suffix

    cypher = in_list_re.sub(_fix_in_list, cypher)
    return cypher


def _sanitize_cypher(text: str) -> str:
    """Lightweight cleanup for occasional model formatting mistakes."""
    if text is None:
        return ""
    s = str(text).strip()
    if "```" in s:
        # Extract first fenced block if present.
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
            # Drop an optional language tag line like "cypher".
            s_lines = [ln.rstrip() for ln in s.splitlines()]
            if s_lines and s_lines[0].strip().lower() in {"cypher", "cql", "neo4j"}:
                s_lines = s_lines[1:]
            s = "\n".join(s_lines).strip()
    s = _unquote_numeric_literals_in_cypher(s)
    return s


@component
class QueryProcessor:

    """Generate a Cypher query for a user question."""

    def __init__(self, model: str | None = None):

        # Prefer specific query model, fallback to general model, then default
        model_name = model or os.getenv(OLLAMA_QUERY_MODEL_ENV) or os.getenv(OLLAMA_MODEL_ENV, DEFAULT_OLLAMA_MODEL)
        ollama_url = os.getenv(OLLAMA_URL_ENV, DEFAULT_OLLAMA_URL)

            # Prompt template
        self.prompt_builder = PromptBuilder(
                        template=("""
You are an expert in Neo4j Cypher.
Your task: produce exactly ONE read-only Cypher query for the user question.

The user questions are in English. Never output German. Never output explanations.

STRICT OUTPUT RULES
1) Output ONLY the Cypher query text (no prose, no labels like “Cypher:”, no comments).
2) Output exactly ONE query (no multiple queries, no separators like ---).
3) Do NOT use parameters ($param). Embed literals directly (numbers unquoted, strings double-quoted).
4) Allowed clauses: MATCH, OPTIONAL MATCH, WHERE, AND, OR, RETURN, ORDER BY, LIMIT.
5) Forbidden: CREATE, MERGE, DELETE, SET, CALL, LOAD CSV, APOC, any writes.
6) Use only these aliases: s, m, e, d, w, h, mi.
7) Do not invent labels/properties/relationships.
8) If you cannot map the question to the schema, return a safe empty result:
     MATCH (m:Measurement) WHERE false RETURN m LIMIT 0

GRAPH SCHEMA (ONLY)
Node labels/properties:
- Sensor { mote_id }
- Measurement { temperature, humidity, light, voltage, time }
- Epoch { epoch_id }
- Date { date }
- Week { week }
- Hour { hour }
- Minute { minute }

Relationships:
- (Sensor)-[:GENERATED]->(Measurement)
- (Measurement)-[:EPOCH_IN]->(Epoch)
- (Measurement)-[:AT_DATE]->(Date)
- (Measurement)-[:AT_WEEK]->(Week)
- (Measurement)-[:AT_HOUR]->(Hour)
- (Measurement)-[:AT_MINUTE]->(Minute)
- (Week)-[:HAS_DATE]->(Date)
- (Date)-[:HAS_HOUR]->(Hour)
- (Hour)-[:HAS_MINUTE]->(Minute)

DOMAIN CONVENTIONS
- Measurement.time is stored like "YYYY-MM-DDTHH:MM:SS..." (may include fractional seconds).
- For questions with a specific timestamp, filter using:
    WHERE m.time STARTS WITH "YYYY-MM-DDTHH:MM:SS"

- Sensor.mote_id is stored as a NUMBER (integer/float). Even if the user writes "1.0",
  treat it as numeric 1.0 and never quote it.

QUERY SHAPES (IMPORTANT)
A) Factual point lookup (sensor + timestamp + metric)
Return exactly: m.time AS time, m.<metric> AS value; ORDER BY m.time ASC; LIMIT 1.

B) List measurements (epoch, optional sensor)
Return exactly these fields:
    coalesce(s.mote_id, m.moteid, m.mote_id) AS mote_id,
    m.time AS time,
    m.temperature AS temperature,
    m.humidity AS humidity,
    m.light AS light,
    m.voltage AS voltage
Order by time, mote_id. LIMIT 20.
If epoch-only: MATCH (m:Measurement) then MATCH (m)-[:EPOCH_IN]->(e:Epoch {epoch_id: <E>}) and OPTIONAL MATCH (s:Sensor)-[:GENERATED]->(m).
If sensor+epoch: MATCH (s:Sensor {mote_id: <S>})-[:GENERATED]->(m:Measurement) then MATCH (m)-[:EPOCH_IN]->(e:Epoch {epoch_id: <E>}).

C) Aggregations (avg/min/max/count) over sensor or epoch
Return exactly one field: <agg>(...) AS value or count(m) AS value.

D) Trend series (sensor + metric)
Return: s.mote_id AS mote_id, m.time AS time, m.<metric> AS value; ORDER BY m.time ASC.

E) Compare trends (two sensors + one metric)
Still output ONE query: use WHERE s.mote_id IN [<S1>, <S2>].
Return mote_id, time, value; ORDER BY time ASC, mote_id ASC.

USER QUESTION:
{{query}}
"""),
            required_variables=["query"]
        )

        # Verbindung zur lokalen Ollama-API
        self.llm = OllamaGenerator(
            model=model_name,
            url=ollama_url,
            generation_kwargs={
                "num_predict": _DEFAULT_QUERY_NUM_PREDICT,
                "temperature": _DEFAULT_QUERY_TEMPERATURE,
            },
        )


    @component.output_types(cypher=str)
    def run(self, query: str):

        """Run the component.

        Args:
            query: Natural-language user question.

        Returns:
            A dict with key `cypher` containing the generated query string.
        """

        if _debug_enabled():
            print("\n[query_processor] input.query:")
            print(query)
        
        # Prompt erstellen
        prompt = self.prompt_builder.run(query=query)["prompt"]

        # LLM-Aufruf über Ollama
        response = self.llm.run(prompt=prompt)

        cypher = _sanitize_cypher(response["replies"][0])

        if _debug_enabled():
            print("[query_processor] output.cypher:")
            print(cypher)

        return {"cypher": cypher}

        












        




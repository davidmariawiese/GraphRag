"""GraphRAG query-to-Cypher component.

This component uses an LLM (Ollama via Haystack integration) to generate exactly
one Cypher query from a natural-language question.
"""

import os
import re

from haystack import component
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator

from ..config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    GRAPH_RAG_DEBUG_ENV,
    OLLAMA_QUERY_NUM_PREDICT_ENV,
    OLLAMA_QUERY_TEMPERATURE_ENV,
    OLLAMA_QUERY_MODEL_ENV,
    OLLAMA_URL_ENV,
    DEFAULT_OLLAMA_QUERY_NUM_PREDICT,
    DEFAULT_OLLAMA_QUERY_TEMPERATURE,
)


def _debug_enabled() -> bool:
    return os.getenv(GRAPH_RAG_DEBUG_ENV) == "true"


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def postprocess_cypher(llm_text: str | None) -> str:
    """Postprocess LLM output into executable Cypher.

    The generator is instructed to output *only* Cypher, but LLMs still
    frequently add formatting that Neo4j can't execute as-is.

    Postprocessing is intentionally conservative and only fixes two common,
    non-semantic issues:

    1) Markdown fenced blocks:
       The model sometimes wraps the query in ```...``` (optionally with a
       language tag). We extract the first fenced block and drop the tag.

    2) Numeric literals quoted as strings:
       Neo4j is type-sensitive. If an ID is stored as a number (e.g. mote_id),
       comparing it to a string like "1.0" yields no matches. We unquote numeric
       strings for known numeric keys and IN-lists on mote_id.
    """

    if llm_text is None:
        return ""

    cypher = str(llm_text).strip()

    # 1) Extract the first fenced code block, if present.
    if "```" in cypher:
        parts = cypher.split("```")
        if len(parts) >= 3:
            cypher = parts[1]
            lines = [ln.rstrip() for ln in cypher.splitlines()]
            if lines and lines[0].strip().lower() in {"cypher", "cql", "neo4j"}:
                lines = lines[1:]
            cypher = "\n".join(lines).strip()

    # 2) Unquote numeric literals for known numeric keys.
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


@component
class QueryProcessor:

    """Generate a Cypher query for a user question."""

    def __init__(self):

        # Prefer query-specific model from env, else fall back to the config default.
        model_name = os.getenv(OLLAMA_QUERY_MODEL_ENV) or DEFAULT_OLLAMA_MODEL
        ollama_url = os.getenv(OLLAMA_URL_ENV, DEFAULT_OLLAMA_URL)

        num_predict = _get_env_int(
            OLLAMA_QUERY_NUM_PREDICT_ENV, DEFAULT_OLLAMA_QUERY_NUM_PREDICT
        )
        temperature = _get_env_float(
            OLLAMA_QUERY_TEMPERATURE_ENV, DEFAULT_OLLAMA_QUERY_TEMPERATURE
        )

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
        self.llm = OllamaGenerator(model=model_name,url=ollama_url,
            generation_kwargs={
                "num_predict": num_predict,
                "temperature": temperature,
            },
        )


    @component.output_types(cypher=str)
    def run(self, query: str) -> dict[str, str]:

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

        cypher = postprocess_cypher(response["replies"][0])

        if _debug_enabled():
            print("[query_processor] output.cypher:")
            print(cypher)

        return {"cypher": cypher}




















        












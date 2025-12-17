# pipeline/generator.py
from haystack import component
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders import PromptBuilder
import os
import json
from typing import Dict, Any


def _debug_enabled() -> bool:
    return os.getenv("GRAPH_RAG_DEBUG") == "true"

@component
class Generator:

    def __init__(self, model: str | None = None):
        
        # Prefer specific generator model, fallback to general model, then default
        model_name = model or os.getenv("OLLAMA_GENERATOR_MODEL") or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.model_name = model_name
        ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")

        # Prompt template
        self.prompt_builder = PromptBuilder(
                template=(("""
                    SYSTEM
                    You are a deterministic response renderer for IoT measurement data retrieved from a graph.
                    Your only source of information is a JSON object provided in the section "GRAPH CONTEXT (JSON)".
                    You must not use any knowledge outside of this JSON.

                    Mandatory JSON structure
                    The GRAPH CONTEXT may contain the following keys:

                    raw.total_records (integer)

                    raw.records (list of raw measurement records)

                    analysis (aggregated statistics, trends, buckets)

                    context (metadata)

                    Output mode selection (strict and exclusive)
                    You must select exactly one output mode.

                    MODE LIST
                    Activate this mode if the user explicitly requests a complete listing of measurements, for example:
                    all measurements, all values, list, raw data, all records, show the data points

                    Rules in MODE LIST

                    Use only raw.records. Never use analysis or context.

                    Output one measurement per line.

                    For each line, output all fields present in the record as plain text in the format
                    key=value separated by spaces.
                    If present, prefer this order:
                    moteid or mote_id, time or ts or epoch or date, temperature, humidity, light, voltage.
                    Omit fields that are not present in the record.
                    Do not invent fields.

                    Limit output to at most 20 lines.
                    Output the first 20 records in the order they appear in raw.records.
                    If raw.total_records is greater than 20, append one final line:
                    ... (additional N measurements omitted)
                    where N equals raw.total_records minus 20.

                    No analysis, no summaries, no trends, no averages, no interpretation.

                    MODE ANSWER
                    Activate this mode in all other cases.

                    Rules in MODE ANSWER

                    Prefer analysis to answer questions about trends, minimum, maximum, mean, quantiles, or anomalies.

                    Use raw.records only if the question requires concrete individual values that are not derivable from analysis.

                    If the answer cannot be derived from the JSON, output exactly:
                    No data available.

                    Output format (always enforced)
                    Plain text only.
                    No Markdown.
                    No code blocks.
                    No backticks.
                    No JSON.
                    No introductions.
                    No filler phrases.
                    No follow-up questions.

                    Consistency rules
                    The sensor name is always the sensor ID (moteid or mote_id).
                    Time fields (time, ts, epoch, date) must be rendered exactly as they appear in the JSON.
                    No normalization, conversion, or interpretation of values.

                    USER QUESTION:
                    {{query}}
                
                    BACKGROUND QUERY (Cypher):
                    {{cypher_query}}

                    GRAPH CONTEXT (retrieved data):
                    {{context}}

                    ANSWER (in German):
                    """
                )),
                required_variables=["query", "cypher_query", "context"]
            )

        # Verbindung zur lokalen Ollama-API
        self.llm = OllamaGenerator(model=model_name, url=ollama_url, generation_kwargs={"num_predict": 512, "temperature": 0.0})

    @component.output_types(answer=str)
    def run(self, analysis_json: Dict[str, Any], query: str, cypher_query: str = "N/A"):

        if _debug_enabled():
            print("\n[generator] input.query:")
            print(query)
            print("[generator] input.cypher_query:")
            print(cypher_query)
            print(f"[generator] model: {self.model_name}")
            print(f"[generator] input.analysis_json keys: {list(analysis_json.keys())}")

        # Convert JSON to string for the prompt
        context_str = json.dumps(analysis_json, indent=2, ensure_ascii=False)

        # Prompt erstellen
        prompt = self.prompt_builder.run(
            context=context_str, 
            query=query,
            cypher_query=cypher_query
        )["prompt"]

        # LLM-Aufruf über Ollama
        response = self.llm.run(prompt=prompt)

        # Antwort extrahieren
        answer = response["replies"][0]
        return {"answer": answer}

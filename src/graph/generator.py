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

        # Prompt-Vorlage
        self.prompt_builder = PromptBuilder(
                template=(("""
                    Du bist ein hilfreicher Assistent für einen IoT-Messdaten-Graphen.
                    Beantworte die Nutzerfrage präzise und kurz auf Deutsch, basierend auf dem folgenden Kontext.
                    Gib ausschließlich Klartext ohne Markdown, ohne Codeblöcke, ohne Backticks, ohne JSON. Keine Aufzählungen in ```...```.

                    REGELN:
                    1. Nutze AUSSCHLIESSLICH die Informationen aus "GRAPH CONTEXT".
                    2. Stelle KEINE Rückfragen, sondern generiere immer eine aussagekräftige natürlichsprachliche Antwort!
                    3. Erfinde nichts; falls die Antwort nicht ableitbar ist, sage: "Dazu liegen mir keine Daten vor.".
                    4. Antworte direkt, ohne Floskeln.
                    5. Nenne konkrete Werte (Temperatur, Zeit, Sensor-ID), wenn vorhanden.
                    6. Sensorname ist die ID (mote_id).
                    7. Wenn nach "allen Messwerten" oder "Liste" gefragt wird und im Kontext `raw.records` vorhanden ist, gib eine kompakte TEXTliste der Messungen (z.B. "- moteid=X time=... temp=... hum=... light=... volt=..."). Kürze bei vielen Einträgen auf ca. 20. KEIN JSON ausgeben.
                     

                    NUTZERFRAGE:
                    {{query}}
                
                    HINTERGRUND-QUERY (Cypher):
                    {{cypher_query}}

                    GRAPH CONTEXT (Gefundene Daten):
                    {{context}}

                    ANTWORT:
                    """
                )),
                required_variables=["query", "cypher_query", "context"]
            )

        # Verbindung zur lokalen Ollama-API
        self.llm = OllamaGenerator(model=model_name, url=ollama_url, generation_kwargs={"num_predict": 256, "temperature": 0.0})

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

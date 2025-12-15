# pipeline/generator.py
from haystack import component
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders import PromptBuilder
import os


def _debug_enabled() -> bool:
    return os.getenv("GRAPH_RAG_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

@component
class Generator:

    def __init__(self, model: str = "llama3.1:8b"):

        # Prompt-Vorlage
        self.prompt_builder = PromptBuilder(
            template=("""
                Du bist ein hilfreicher Assistent für einen IoT-Messdaten-Graphen.
                Deine Aufgabe ist es, die Nutzerfrage präzise und natürlich auf Deutsch zu beantworten, basierend auf den bereitgestellten Daten.

                REGELN:
                1. Nutze AUSSCHLIESSLICH die Informationen aus dem Abschnitt "GRAPH CONTEXT".
                2. Erfinde keine Daten. Wenn der Kontext leer ist oder die Antwort nicht enthält, sage: "Dazu liegen mir keine Daten vor."
                3. Bitte antworte neutral und sachlich
                4. Nenne konkrete Werte (Temperatur, Zeit, Sensor-ID), wenn danach gefragt wird.
                5. Antworte immer auf Deutsch.
                6. Der Sensorname ist eine ID(mote_id)

                NUTZERFRAGE:
                {{query}}
                
                HINTERGRUND-QUERY (Cypher):
                {{cypher_query}}

                GRAPH CONTEXT (Gefundene Daten):
                {{context}}

                ANTWORT:
                """
            ),
            required_variables=["query", "cypher_query", "context"]
        )

        # Verbindung zur lokalen Ollama-API
        self.llm = OllamaGenerator(model=model, url="http://ollama:11434", generation_kwargs={"num_predict": 256, "temperature": 0.0})

    @component.output_types(answer=str)
    def run(self, context: str, query: str, cypher_query: str = "N/A"):

        if _debug_enabled():
            print("\n[generator] input.query:")
            print(query)
            print("[generator] input.cypher_query:")
            print(cypher_query)
            ctx = context or ""
            preview = ctx[:800].rstrip()
            suffix = "..." if len(ctx) > 800 else ""
            print(f"[generator] input.context_preview (len={len(ctx)}):")
            print(preview + suffix)

        # Prompt erstellen
        prompt = self.prompt_builder.run(
            context=context, 
            query=query,
            cypher_query=cypher_query
        )["prompt"]

        # print("[GENERATOR]:")
        # print(prompt)
        # LLM-Aufruf über Ollama
        response = self.llm.run(prompt=prompt)

        # Antwort extrahieren
        answer = response["replies"][0]
        return {"answer": answer}

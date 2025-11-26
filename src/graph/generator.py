# pipeline/generator.py
from haystack import component
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders import PromptBuilder

@component
class Generator:

    def __init__(self, model: str = "llama3.1:8b"):

        # Prompt-Vorlage
        self.prompt_builder = PromptBuilder(
            template=("""
                You are a highly reliable assistant answering questions strictly from a Neo4j knowledge graph.

                RULES (must follow):
                1. You must only use the information contained in the provided graph data (documents).
                2. If information is missing, incomplete, or ambiguous, explicitly state that it is not available.
                3. Do not infer or guess any values, relationships, or interpretations that cannot be directly confirmed.
                4. Do not fabricate measurements, timestamps, or sensor attributes.
                5. When measurements are shown, interpret them exactly as given.
                6. Always separate what "is in the graph" from what "is not in the graph".

                GRAPH SCHEMA (for context):
                Nodes:
                - Sensor, Measurement, Epoch, Date, Week, Hour, Minute

                Edges:
                - GENERATED, EPOCH_IN, AT_DATE, AT_WEEK, AT_HOUR, AT_MINUTE, 
                HAS_DATE, HAS_HOUR, HAS_MINUTE

                QUESTION:
                {{query}}

                GRAPH CONTEXT:
                {{context}}

                ANSWER:
                Provide an accurate answer based strictly on the graph context. If insufficient data is available, say so clearly.
                """
            )
        )

        # Verbindung zur lokalen Ollama-API
        self.llm = OllamaGenerator(model=model, url="http://ollama:11434", generation_kwargs={"num_predict": 256, "temperature": 0.3})

    @component.output_types(answer=str)
    def run(self, context: str, query: str):

        # Prompt erstellen
        prompt = self.prompt_builder.run(
            context=context, 
            query=query
        )["prompt"]

        # LLM-Aufruf über Ollama
        response = self.llm.run(prompt=prompt)

        # Antwort extrahieren
        answer = response["replies"][0]
        return {"answer": answer}

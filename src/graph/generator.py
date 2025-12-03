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
                Du bist ein hilfreicher Assistent für einen IoT-Messdaten-Graphen.
                Deine Aufgabe ist es, die Nutzerfrage präzise und natürlich auf Deutsch zu beantworten, basierend auf den bereitgestellten Daten.

                REGELN:
                1. Nutze AUSSCHLIESSLICH die Informationen aus dem Abschnitt "GRAPH CONTEXT".
                2. Erfinde keine Daten. Wenn der Kontext leer ist oder die Antwort nicht enthält, sage: "Dazu liegen mir keine Daten vor."
                3. Fasse die Messwerte sinnvoll zusammen (z.B. "Es gibt 5 Messungen im Zeitraum X" oder "Die Temperatur lag zwischen A und B").
                4. Nenne konkrete Werte (Temperatur, Zeit, Sensor-ID), wenn danach gefragt wird.
                5. Antworte immer auf Deutsch.
                6. Der Sensorname ist eine ID(mote_id)

                NUTZERFRAGE:
                {{query}}

                GRAPH CONTEXT (Gefundene Daten):
                {{context}}

                ANTWORT:
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

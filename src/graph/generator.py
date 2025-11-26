# pipeline/generator.py
from haystack import component
from haystack.components.generators import OllamaGenerator
from haystack.components.builders import PromptBuilder

@component
class Generator:

    def __init__(self, model: str = "llama3"):

        # Prompt-Vorlage
        self.prompt_builder = PromptBuilder(
            template=(
                "You are an expert IoT and Sensor-Data assistant.\n"
                "Answer the question strictly based on the following context.\n\n"
                "Context:\n{{context}}\n\n"
                "Question: {{query}}\n"
            )
        )

        # Verbindung zur lokalen Ollama-API
        self.llm = OllamaGenerator(model=model, url="http://ollama:11434", temperature=0.2)

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

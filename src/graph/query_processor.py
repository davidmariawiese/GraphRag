from haystack import component


@component
class QueryProcessor:
    """
    Nimmt eine natürliche Sprachfrage und erzeugt eine Cypher-Query als Text.
    Hier ist der LLM-Aufruf als Platzhalter implementiert – du kannst ihn später
    durch einen echten Generator (z.B. OpenAI, Ollama-Connector) ersetzen.
    """

    def __init__(self):
        # hier später: self.generator = OllamaGenerator(...) oder ähnliches
        ...

    @component.output_types(cypher=str, raw_output=str)
    def run(self, query: str):
        # TODO: hier echten LLM-Aufruf einbauen
        # Für den Start: einfache, feste Demo-Query zurückgeben
        demo_cypher = "MATCH (m:Measurement) RETURN m LIMIT 5"

        # In einem echten Setup würdest du z.B. so etwas tun:
        # llm_response = self.generator.run(prompt=PROMPT.format(query=query))
        # raw_output = llm_response["replies"][0]
        raw_output = f'{{"cypher": "{demo_cypher}"}}'

        return {
            "cypher": demo_cypher,
            "raw_output": raw_output,
        }
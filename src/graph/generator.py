from typing import List
from haystack import component, Document


DEFAULT_GENERATOR_PROMPT = """
Nutze den folgenden Graph-Kontext, um die Nutzerfrage präzise zu beantworten:

{context}

Nutzerfrage: {query}

Antwort:
"""


@component
class Generator:
    """
    Nimmt eine Liste von Kontext-Dokumenten und eine Frage
    und erzeugt eine natürliche Sprache Antwort.
    """

    def __init__(self):
        # hier später: self.generator = OllamaGenerator(...) oder anderer LLM-Connector
        ...

    @component.output_types(answer=str)
    def run(self, documents: List[Document], query: str):
        # Kontext zusammenbauen
        context_snippets = [doc.content for doc in documents]
        context = "\n\n".join(context_snippets)

        prompt = DEFAULT_GENERATOR_PROMPT.format(context=context, query=query)

        # TODO: Echten LLM-Aufruf einbauen
        # reply = self.generator.run(prompt=prompt)["replies"][0]
        reply = (
            "Dies ist eine Platzhalter-Antwort. "
            "Hier würde der LLM basierend auf dem Kontext antworten."
        )

        return {"answer": reply}
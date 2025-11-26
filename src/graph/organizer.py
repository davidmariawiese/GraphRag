# pipeline/organizer.py
from haystack import component

@component
class Organizer:

    @component.output_types(context=str)
    def run(self, documents: list):
        text = "\n".join([d.content for d in documents])
        return {"context": text}

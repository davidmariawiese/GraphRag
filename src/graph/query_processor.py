from haystack import component

@component
class QueryProcessor:

    @component.output_types(cypher=str)
    def run(self, query: str):
        # Hier später Ihre LLM-Query-Planung einfügen
        cypher = f"""
        MATCH (m:MEASUREMENT)
        RETURN m
        LIMIT 10
        """
        return {"cypher": cypher}

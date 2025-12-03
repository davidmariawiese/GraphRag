from haystack import component
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders import PromptBuilder

@component
class QueryProcessor:

    def __init__(self, model: str = "llama3.1:8b"):

        # Prompt-Vorlage
        self.prompt_builder = PromptBuilder(
            template=("""
                        You are an expert in Neo4j Cypher. Your task is to generate exactly one valid Cypher query based on the user's question.

                        STRICT RULES:
                        1. Only produce a single Cypher query.
                        2. Use only MATCH, OPTIONAL MATCH, WHERE, AND, OR, RETURN, ORDER BY, LIMIT.
                        3. Do not use CREATE, MERGE, DELETE, SET, CALL, LOAD CSV, APOC, or any write operations.
                        4. Do not add explanations, comments, natural language, or code fences.
                        5. Do not wrap the query in ```cypher``` or any other code block.
                        6. Always use short lowercase aliases (s, m, e, d, w, h, mi).
                        7. Do not invent labels, properties, or relationships that are not listed below.
                        8. If the question is ambiguous, choose the simplest correct interpretation.

                        GRAPH SCHEMA:
                        Node labels and properties:
                        - Sensor { mote_id }
                        - Measurement { temperature, humidity, light, voltage, time }
                        - Epoch { epoch_id }
                        - Date { date }
                        - Week { week }
                        - Hour { hour }
                        - Minute { minute }

                        Relationships:
                        - (Sensor)-[:GENERATED]->(Measurement)
                        - (Measurement)-[:EPOCH_IN]->(Epoch)
                        - (Measurement)-[:AT_DATE]->(Date)
                        - (Measurement)-[:AT_WEEK]->(Week)
                        - (Measurement)-[:AT_HOUR]->(Hour)
                        - (Measurement)-[:AT_MINUTE]->(Minute)
                        - (Week)-[:HAS_DATE]->(Date)
                        - (Date)-[:HAS_HOUR]->(Hour)
                        - (Hour)-[:HAS_MINUTE]->(Minute)

                        REQUIREMENTS:
                        - Always return relevant nodes or fields.
                        - Use DISTINCT when returning nodes that can appear multiple times.
                        - Do not hallucinate any label, property, or relationship.
                        - Prefer direct relationships from Measurement when possible (e.g., AT_DATE over walking through Week → Date).
                        - Only use WHERE for comparisons or filters taken from the user’s question.
                      
                        EXAMPLES (FEW-SHOT):
                      
                        User question:
                        Which sensors measured in epoch 2?
                        Cypher:
                        MATCH (s:Sensor)-[:GENERATED]->(m:Measurement)-[:EPOCH_IN]->(e:Epoch {epoch_id: 2})
                        RETURN DISTINCT s,e

                        User question:
                        Show all Measurements of sensor 12 on date "2025-01-03".
                        Cypher:
                        MATCH (s:Sensor {mote_id: 12})-[:GENERATED]->(m:Measurement)-[:AT_DATE]->(d:Date {date: "2025-01-03"})
                        RETURN m,s

                        User question:
                        Which hours and minutes belong to epoch 1?
                        Cypher:
                        MATCH (m:Measurement)-[:EPOCH_IN]->(e:Epoch {epoch_id: 1})
                        MATCH (m)-[:AT_HOUR]->(h:Hour)
                        MATCH (m)-[:AT_MINUTE]->(mi:Minute)
                        RETURN DISTINCT h, mi


                        USER QUESTION:
                        {{query}}

                        """
            )
        )

        # Verbindung zur lokalen Ollama-API
        self.llm = OllamaGenerator(model=model, url="http://ollama:11434", generation_kwargs={"num_predict": 128, "temperature": 0.0})


    @component.output_types(cypher=str)
    def run(self, query: str):
        
        # Prompt erstellen
        prompt = self.prompt_builder.run(query=query)["prompt"]

        # LLM-Aufruf über Ollama
        response = self.llm.run(prompt=prompt)

        #Cypher Ausgeben
        print("[QUERY_PROCESSOR:]")
        print(response)

        # Antwort extrahieren
        answer = response["replies"][0]
        return {"cypher": answer}









        

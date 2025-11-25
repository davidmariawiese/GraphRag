from haystack_integrations.components.generators.ollama import OllamaGenerator

from haystack import Pipeline, Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore

def create_rag_pipeline(documents, ollama_model="llama3", ollama_url="http://ollama:11434", generation_kwargs={"num_predict": 100, "temperature": 0.9}):
  """
  Erstellt eine einfache RAG-Pipeline mit BM25-Retriever und Ollama-Generator.
  """
  docstore = InMemoryDocumentStore()
  docstore.write_documents(documents)

  generator = OllamaGenerator(model=ollama_model, url=ollama_url, generation_kwargs=generation_kwargs)
  retriever = InMemoryBM25Retriever(document_store=docstore)
  template = """
      Given the following information, answer the question.

      Context:
      {% for document in documents %}
          {{ document.content }}
      {% endfor %}

      Question: {{query}}
      Answer:
      """
    
  prompt_builder = PromptBuilder(template=template)

  pipeline = Pipeline()
  pipeline.add_component("retriever", retriever)
  pipeline.add_component("prompt_builder", prompt_builder)
  pipeline.add_component("llm", generator)
  pipeline.connect("retriever", "prompt_builder.documents")
  pipeline.connect("prompt_builder", "llm")

  return pipeline

def run_rag_pipeline(pipe, query):
  result = pipe.run({"prompt_builder": {"query": query},"retriever": {"query": query}})
  return result
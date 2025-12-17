import json
from pathlib import Path
from typing import List, Optional

from haystack_integrations.components.generators.ollama import OllamaGenerator

from haystack import Pipeline, Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore


def load_base_documents(data_path: Optional[Path] = None) -> List[Document]:
  """Load documents from the base JSONL dataset."""

  path = Path(data_path) if data_path else Path(__file__).resolve().parents[2] / "data" / "base" / "data.jsonl"
  with path.open("r", encoding="utf-8") as handle:
    entries = [json.loads(line) for line in handle]

  return [Document(content=entry.get("content", ""), meta=entry.get("meta", {})) for entry in entries]

def create_rag_pipeline(documents: Optional[List[Document]] = None, ollama_model: str = "llama3", ollama_url: str = "http://ollama:11434", generation_kwargs={"num_predict": 100, "temperature": 0.9}):
  """Erstellt eine einfache RAG-Pipeline mit BM25-Retriever und Ollama-Generator und lädt standardmäßig die Basisdaten."""

  docs = documents or load_base_documents()

  docstore = InMemoryDocumentStore()
  docstore.write_documents(docs)

  generator = OllamaGenerator(model=ollama_model, url=ollama_url, generation_kwargs=generation_kwargs)
  retriever = InMemoryBM25Retriever(document_store=docstore)
  template = """
    You are a helpful assistant for IoT measurement data. Answer briefly and precisely IN GERMAN.

    RULES:
    1. Use ONLY the information from the "KONTEXT" section.
    2. Do not invent any data. If the context does not allow an answer, reply: "Dazu liegen mir keine Daten vor.".
    3. Answer directly, without filler phrases.
    4. Include concrete values (temperature, time, sensor id) when available.
    5. The sensor name is the id (mote_id).

    KONTEXT:
    {% for document in documents %}
      - {{ document.content }}
    {% endfor %}

    QUESTION: {{query}}

    ANSWER (in German):
    """

  prompt_builder = PromptBuilder(template=template, required_variables=["documents", "query"])

  pipeline = Pipeline()
  pipeline.add_component("retriever", retriever)
  pipeline.add_component("prompt_builder", prompt_builder)
  pipeline.add_component("llm", generator)
  pipeline.connect("retriever", "prompt_builder.documents")
  pipeline.connect("prompt_builder", "llm")

  return pipeline

def run_rag_pipeline(pipe, query: str, debug: bool = False, preview_docs: int = 3):
  result = pipe.run({"prompt_builder": {"query": query},"retriever": {"query": query}}, include_outputs_from=["retriever", "prompt_builder"])

  if debug:
    print(f"[DEBUG] Pipeline keys: {list(result.keys())}")

    retriever_output = result.get("retriever")
    docs = []

    if isinstance(retriever_output, dict):
      docs = retriever_output.get("documents", [])
    elif isinstance(retriever_output, list):
      docs = retriever_output

    # Fall back to prompt_builder output if retriever is missing
    if not docs:
      pb_output = result.get("prompt_builder", {})
      if isinstance(pb_output, dict):
        docs = pb_output.get("documents", []) or docs

    print(f"[DEBUG] Retrieved {len(docs)} documents")
    for idx, doc in enumerate(docs[:preview_docs]):
      print(f"[DEBUG] Doc {idx+1} content: {doc.content[:200]}")
      print(f"[DEBUG] Doc {idx+1} meta: {doc.meta}")

  return result
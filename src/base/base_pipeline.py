"""Base RAG pipeline (BM25 + Ollama) over a local JSONL corpus.

This module is used directly by notebooks (see `notebooks/04_base_pipeline_test.ipynb`).

Design goals:
- Keep behavior stable and transparent.
- Prefer environment variables for configuration (typically loaded via `.env`).
- Keep I/O at the edges: dataset loading and debug printing are isolated.

Inputs:
- Dataset: `data/base/data.jsonl` (JSONL; one JSON object per line).
- Env vars: Ollama URL/model selection.

Outputs:
- A Haystack `Pipeline` and raw pipeline run outputs.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from haystack_integrations.components.generators.ollama import OllamaGenerator

from haystack import Pipeline, Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore

from ..config import (
  DEFAULT_OLLAMA_MODEL,
  DEFAULT_OLLAMA_URL,
  OLLAMA_GENERATOR_MODEL_ENV,
  OLLAMA_MODEL_ENV,
  OLLAMA_URL_ENV,
)


load_dotenv()


_DEFAULT_BASE_DATASET_RELATIVE = Path("data") / "base" / "data.jsonl"
_DEFAULT_GENERATION_KWARGS: Dict[str, Any] = {"num_predict": 100, "temperature": 0.9}
_DEFAULT_PREVIEW_DOCS = 3
_DEFAULT_RETRIEVER_TOP_K = 50


_RE_SENSOR = re.compile(r"\bsensor\s*(\d+(?:\.0)?)\b", re.IGNORECASE)
_RE_EPOCH = re.compile(r"\bepoch\s*(\d+)\b", re.IGNORECASE)
_RE_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_RE_TIME = re.compile(r"\b(\d{2}:\d{2}:\d{2})\b")


def load_base_documents(data_path: Optional[Path] = None) -> List[Document]:
  """Load documents from the base JSONL dataset.

  Args:
    data_path: Optional explicit path to a JSONL file. If omitted, defaults to
      `data/base/data.jsonl` relative to the repository root.

  Returns:
    A list of Haystack `Document` objects.
  """

  path = Path(data_path) if data_path else Path(__file__).resolve().parents[2] / _DEFAULT_BASE_DATASET_RELATIVE
  with path.open("r", encoding="utf-8") as handle:
    entries = [json.loads(line) for line in handle]

  return [Document(content=entry.get("content", ""), meta=entry.get("meta", {})) for entry in entries]

def create_rag_pipeline(
  documents: Optional[List[Document]] = None,
  ollama_model: Optional[str] = None,
  ollama_url: Optional[str] = None,
  generation_kwargs: Optional[Dict[str, Any]] = None,
):
  """Create a simple RAG pipeline (BM25 + Ollama) and load base data by default.

  Configuration resolution (kept stable):
  - `ollama_url` argument overrides `OLLAMA_URL`, else defaults to `http://ollama:11434`.
  - `ollama_model` argument overrides `OLLAMA_GENERATOR_MODEL` then `OLLAMA_MODEL`, else defaults to `llama3.1:8b`.
  - `generation_kwargs` argument overrides defaults; default is non-deterministic
    (`temperature=0.9`) to preserve existing behavior.

  Args:
    documents: Optional document list to index. If omitted/falsey, base documents
      are loaded from disk.
    ollama_model: Optional model name override.
    ollama_url: Optional Ollama base URL override.
    generation_kwargs: Optional generation parameters for the Ollama generator.

  Returns:
    A configured Haystack `Pipeline`.
  """

  resolved_url = ollama_url or os.getenv(OLLAMA_URL_ENV, DEFAULT_OLLAMA_URL)
  resolved_model = ollama_model or os.getenv(OLLAMA_GENERATOR_MODEL_ENV) or os.getenv(OLLAMA_MODEL_ENV, DEFAULT_OLLAMA_MODEL)
  resolved_kwargs = generation_kwargs or _DEFAULT_GENERATION_KWARGS

  docs = documents or load_base_documents()

  docstore = InMemoryDocumentStore()
  docstore.write_documents(docs)

  generator = OllamaGenerator(model=resolved_model, url=resolved_url, generation_kwargs=resolved_kwargs)
  retriever = InMemoryBM25Retriever(document_store=docstore)
  template = """
    You are a helpful assistant for IoT measurement data. Answer briefly and precisely IN ENGLISH.

    RULES:
    1. Use ONLY the information from the "KONTEXT" section.
    2. Do not invent any data. If the context does not allow an answer, reply EXACTLY: "No data available.".
    3. Answer directly, without filler phrases.
    4. Include concrete values (temperature, time, sensor id) when available.
    5. The sensor name is the id (mote_id).

    KONTEXT:
    {% for document in documents %}
      - {{ document.content }}
    {% endfor %}

    QUESTION: {{query}}

    ANSWER (in English):
    """

  prompt_builder = PromptBuilder(template=template, required_variables=["documents", "query"])

  pipeline = Pipeline()
  pipeline.add_component("retriever", retriever)
  pipeline.add_component("prompt_builder", prompt_builder)
  pipeline.add_component("llm", generator)
  pipeline.connect("retriever", "prompt_builder.documents")
  pipeline.connect("prompt_builder", "llm")

  return pipeline

def run_rag_pipeline(pipe, query: str, debug: bool = False, preview_docs: int = _DEFAULT_PREVIEW_DOCS):
  """Run the pipeline for a query.

  Args:
    pipe: The Haystack pipeline created via `create_rag_pipeline`.
    query: User question.
    debug: If true, prints a short, deterministic preview of retrieved documents.
    preview_docs: Number of retrieved documents to print in debug mode.

  Returns:
    The raw Haystack pipeline output dictionary.
  """

  result = pipe.run(
    {
      "prompt_builder": {"query": query},
      "retriever": {"query": query, "top_k": _DEFAULT_RETRIEVER_TOP_K},
    },
    include_outputs_from=["retriever", "prompt_builder"],
  )

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
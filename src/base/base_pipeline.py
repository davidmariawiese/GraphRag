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
  OLLAMA_URL_ENV,
  OLLAMA_GENERATOR_NUM_PREDICT_ENV,
  OLLAMA_GENERATOR_TEMPERATURE_ENV,
  DEFAULT_OLLAMA_GENERATOR_NUM_PREDICT,
  DEFAULT_OLLAMA_GENERATOR_TEMPERATURE,
)


load_dotenv()


_DEFAULT_BASE_DATASET_RELATIVE = Path("data") / "base" / "data.jsonl"
_DEFAULT_PREVIEW_DOCS = 3
_DEFAULT_RETRIEVER_TOP_K = 50




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
  - `ollama_model` argument overrides `OLLAMA_GENERATOR_MODEL`, else falls back to the config default.
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
  resolved_model = (
    ollama_model
    or os.getenv(OLLAMA_GENERATOR_MODEL_ENV)
    or DEFAULT_OLLAMA_MODEL
  )
  if generation_kwargs:
    resolved_kwargs = generation_kwargs
  else:
    # Read generation kwargs from env if present, else use defaults from config
    try:
      num_predict = int(os.getenv(OLLAMA_GENERATOR_NUM_PREDICT_ENV, DEFAULT_OLLAMA_GENERATOR_NUM_PREDICT))
    except Exception:
      num_predict = DEFAULT_OLLAMA_GENERATOR_NUM_PREDICT
    try:
      temperature = float(os.getenv(OLLAMA_GENERATOR_TEMPERATURE_ENV, DEFAULT_OLLAMA_GENERATOR_TEMPERATURE))
    except Exception:
      temperature = DEFAULT_OLLAMA_GENERATOR_TEMPERATURE
    resolved_kwargs = {"num_predict": num_predict, "temperature": temperature}

  docs = documents or load_base_documents()

  docstore = InMemoryDocumentStore()
  docstore.write_documents(docs)

  generator = OllamaGenerator(model=resolved_model, url=resolved_url, generation_kwargs=resolved_kwargs)
  retriever = InMemoryBM25Retriever(document_store=docstore)
  template = """
    You answer questions about IoT measurements.
    Your response is evaluated by EXACT STRING MATCH against a gold label.

     CRITICAL RULES (must follow):
     1) Use ONLY the information in the CONTEXT.
     2) Do NOT guess, extrapolate, or assume the context is complete.
       The retrieved context may be a subset; never infer totals/averages from partial context.
     3) Output MUST be the final answer only: no preamble, no explanations, no bullets, no markdown, no quotes.
       Do not write words like "Based on", "However", "To determine", or similar.
       Do NOT round numbers. Do NOT add or remove decimals.
     4) If the context does not contain enough information to answer EXACTLY, output EXACTLY:
       No data available.

     OUTPUT FORMATS (match EXACTLY):

    A) Factual single value at a time
       <Metric> of sensor <sensor_id> at <YYYY-MM-DD> <HH:MM:SS>: <value><unit>
       - Metric is one of: Temperature | Humidity | Light | Voltage
       - Units: Temperature uses °C, Humidity uses %, Light uses Lux, Voltage uses V
       - No space before the unit (e.g., 19.7336°C, 39.62%, 50.6Lux, 2.68742V)
       - Matching rule for timestamps:
         If the question asks for <HH:MM:SS>, then ANY context timestamp that starts with that prefix
         (e.g., 00:58:46.002832) is an EXACT match.
       - If context times include fractions, truncate to seconds in the output.
       - Extract the value from the matching context line and keep full precision.

      Examples:
      Temperature of sensor 19.0 at 2004-02-28 00:58:46: 19.7336°C
      Voltage of sensor 34.0 at 2004-02-28 00:58:47: 2.68742V

     B) List measurements
       One measurement per line, with this exact key order:
       mote_id=<id> time=<YYYY-MM-DD>T<HH:MM:SS(.fraction)> temperature=<temp> humidity=<hum> light=<light> voltage=<volt>
       - Use values exactly as they appear in context (no units).
       - If more than 20 matches exist in context, output only the first 20 lines.

     C) Aggregations
       - Average Temperature (Epoch <epoch_id>): <value>°C
       - Maximum Light (Epoch <epoch_id>): <value>Lux
       - Number of measurements (Sensor <sensor_id>): <integer>
       - Minimum Voltage (Sensor <sensor_id>): <value>V

      Examples:
      Number of measurements (Sensor 48.0): 1775
      Minimum Voltage (Sensor 31.0): 2.06222V
      Average Temperature (Epoch 2): 54.650252173913046°C
      Maximum Light (Epoch 2): 1847.36Lux

     D) Trend
       The trend of Temperature for sensor <sensor_id> is: <increasing|decreasing|flat>.
       - Decide trend ONLY if you can order at least 2 temperature measurements by time for that sensor from context.
       - Compare earliest vs latest temperature: higher => increasing, lower => decreasing, equal => flat.

      Example:
      The trend of Temperature for sensor 31.0 is: increasing.

     CONTEXT:
     {% for document in documents %}
     {{ document.content }}
     {% endfor %}

     QUESTION: {{query}}

     ANSWER:
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
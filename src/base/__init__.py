"""Base (non-graph) RAG pipeline.

Public entry points:
- `create_rag_pipeline`
- `run_rag_pipeline`
- `load_base_documents`

The implementation lives in `base_pipeline.py` and is imported by notebooks.
"""

from .base_pipeline import create_rag_pipeline, load_base_documents, run_rag_pipeline

__all__ = [
    "create_rag_pipeline",
    "load_base_documents",
    "run_rag_pipeline",
]

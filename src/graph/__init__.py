"""GraphRAG pipeline components.

Public entry points:
- `build_pipeline`
- `run_pipeline`

The implementation lives in `graph_pipeline.py` and component modules.
"""

from .graph_pipeline import build_pipeline, run_pipeline

__all__ = [
    "build_pipeline",
    "run_pipeline",
]

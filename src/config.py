"""Central configuration constants.

This module intentionally contains *only* environment variable keys and default values.
It exists to keep configuration usage consistent across modules without changing any
runtime behavior (no parsing, no validation).

The project config is primarily controlled via environment variables loaded
from a .env file in the repository root
"""

# Debug
GRAPH_RAG_DEBUG_ENV: str = "GRAPH_RAG_DEBUG"



# Ollama
OLLAMA_URL_ENV: str = "OLLAMA_URL"
DEFAULT_OLLAMA_URL: str = "http://ollama:11434"

DEFAULT_OLLAMA_MODEL: str = "llama3.1:8b"

OLLAMA_QUERY_MODEL_ENV: str = "OLLAMA_QUERY_MODEL"
OLLAMA_GENERATOR_MODEL_ENV: str = "OLLAMA_GENERATOR_MODEL"

# Ollama generation parameters
OLLAMA_QUERY_NUM_PREDICT_ENV: str = "OLLAMA_QUERY_NUM_PREDICT"
DEFAULT_OLLAMA_QUERY_NUM_PREDICT: int = 256

OLLAMA_QUERY_TEMPERATURE_ENV: str = "OLLAMA_QUERY_TEMPERATURE"
DEFAULT_OLLAMA_QUERY_TEMPERATURE: float = 0.0

OLLAMA_GENERATOR_NUM_PREDICT_ENV: str = "OLLAMA_GENERATOR_NUM_PREDICT"
DEFAULT_OLLAMA_GENERATOR_NUM_PREDICT: int = 512

OLLAMA_GENERATOR_TEMPERATURE_ENV: str = "OLLAMA_GENERATOR_TEMPERATURE"
DEFAULT_OLLAMA_GENERATOR_TEMPERATURE: float = 0.0

# Neo4j
NEO4J_URI_ENV: str = "NEO4J_URI"
DEFAULT_NEO4J_URI: str = "bolt://neo4j:7687"

NEO4J_USER_ENV: str = "NEO4J_USER"
DEFAULT_NEO4J_USER: str = "neo4j"

NEO4J_PASSWORD_ENV: str = "NEO4J_PASSWORD"
DEFAULT_NEO4J_PASSWORD: str = "password"

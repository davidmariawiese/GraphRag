# GraphRAG for Smart Building Sensor Data

This repository contains the implementation of a **graph-based Retrieval-Augmented Generation (GraphRAG) pipeline**
for natural language querying of **Smart Building / IoT sensor data**.
The project was developed as part of a seminar paper in the context of Design Science Research.

The goal of the project is to demonstrate how **graph-based retrieval** can improve answer quality
for time-dependent, structured sensor data compared to a classical text-based RAG baseline.

---

## Project Overview

The repository implements two RAG pipelines:

1. **GraphRAG Pipeline**
   - Natural language queries are translated into Cypher queries.
   - Sensor data is retrieved from a Knowledge Graph (Neo4j).
   - Results are deterministically analyzed and structured.
   - A Large Language Model generates answers strictly grounded in retrieved data.

2. **TextRAG Baseline**
   - Classical BM25-based retrieval over textually serialized sensor data.
   - Identical LLM, prompts, and generation parameters.
   - Used for controlled comparison in the evaluation.

Both pipelines use the **same LLM configuration** to ensure fair comparison.


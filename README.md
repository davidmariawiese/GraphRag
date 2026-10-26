# GraphRAG for Smart Building Sensor Data

Natural-language question answering over IoT sensor data with a **graph-based Retrieval-Augmented Generation (GraphRAG)** pipeline, evaluated against a classical text-based RAG baseline.

> Seminar project, B.Sc. Computational Visualistics, University of Koblenz (2025) — developed following the Design Science Research approach (Peffers et al.).

**Result:** On 100 test questions with deterministic gold labels, GraphRAG reaches a token-level macro-F1 of **0.726**, compared to **0.150** for a BM25 TextRAG baseline using the same LLM, prompts and generation parameters.

\---

## Why GraphRAG for sensor data?

Classical RAG pipelines are built for unstructured text. When structured, time-dependent sensor readings are serialized into text, timestamps, sensor relations and numeric context get lost — and the LLM starts guessing.

This project keeps the structure: sensor readings live in a **Neo4j knowledge graph**, the question is translated into a **Cypher query**, and the LLM only phrases an answer from deterministically retrieved and analysed data.

## Architecture

```
Question ──► Query Processor ──► Graph Retriever ──► Organizer ──► Generator ──► Answer
             (LLM: NL → Cypher)  (Neo4j)            (deterministic   (LLM, answers only
                                                     statistics)      from context)
```

|Component|What it does|
|-|-|
|**Query Processor**|An LLM translates the question into a Cypher query using a system prompt with the graph schema; the query is then validated and post-processed by rules (schema conformity, forbidden clauses).|
|**Graph Retriever**|Executes the Cypher query against Neo4j and normalizes the result to JSON (one row per measurement). No generative parts.|
|**Organizer**|Deterministic analysis per sensor and metric: global statistics, linear-regression trend, adaptive time buckets, z-score anomaly flags; capped raw rows for list questions; detects aggregate queries.|
|**Generator**|Produces the final answer strictly from the organizer's context with fixed output formats; returns `No data available` if the context is insufficient.|
|**TextRAG baseline**|BM25 retrieval (`top\\\_k = 50`) over one text line per measurement, with the identical generator — the only difference is knowledge representation and retrieval.|

## Results

Evaluation: 100 questions, gold answers generated deterministically from the dataset via reference queries, token-level F1 after normalization (lowercase, no punctuation), averaged over all questions (macro-F1). Temperature 0.0, single run.

|Question type|Questions|GraphRAG F1|TextRAG F1|
|-|-:|-:|-:|
|Factual (single value)|40|0.849|0.243|
|Aggregation (min/max/mean)|30|0.891|0.087|
|List|20|0.329|0.090|
|Trend|5|0.071|0.110|
|Negative (no data)|5|1.000|0.067|
|**Overall (macro-F1)**|**100**|**0.726**|**0.150**|

GraphRAG is clearly stronger on factual, aggregation and negative questions, where exact filtering by sensor, time and epoch matters. List and trend questions remain weak spots — trend answers are free-form and penalized by the token-level metric.

**Limitations:** one dataset, 100 questions, automatic token-level metric only (semantically equivalent but differently phrased answers are penalized); no statements on latency or scalability.

## Tech stack

Python 3.11 · [Haystack 2](https://haystack.deepset.ai/) · Neo4j 5 / Cypher · [Ollama](https://ollama.com/) (local LLMs, evaluated with `qwen2.5:32b-instruct`) · BM25 · pandas · Docker Compose · Jupyter

## Dataset

[Intel Berkeley Research Lab Sensor Data](http://db.csail.mit.edu/labdata/labdata.html): 54 sensors measuring temperature, humidity, light and voltage. This project uses the first 2,000 epochs.

The raw file (`data.txt`, \~150 MB) is **not included**. The preprocessed graph import (`data/graph/data.csv`) and the baseline corpus (`data/base/data.jsonl`) are included, so you can skip preprocessing.

To rebuild from scratch, download `data.txt.gz` from the link above, unpack it to `data/raw/data.txt` and run notebook `01\\\_preprocessing.ipynb`.

## Getting started

Requirements: Docker with Docker Compose. Ollama is configured for an NVIDIA GPU; the evaluated 32B model needs a GPU with plenty of VRAM — smaller models can be set in `.env`.

```bash
git clone https://github.com/davidmariawiese/GraphRag.git
cd GraphRag
cp .env.example .env                       # adjust models if needed
docker compose -f docker/docker-compose.yml up -d
docker exec ollama ollama pull qwen2.5:32b-instruct
```

Then open Jupyter at http://localhost:8888 and run the notebooks in order:

|Notebook|Purpose|
|-|-|
|`01\\\_preprocessing`|Raw data → CSV (graph) and JSONL (baseline)|
|`02\\\_graph\\\_construction`|Builds the knowledge graph in Neo4j|
|`03\\\_graphRag\\\_pipeline\\\_test`|Try the GraphRAG pipeline with your own questions|
|`04\\\_base\\\_pipeline\\\_test`|Try the TextRAG baseline|
|`05\\\_generate\\\_goldlabels`|Creates gold answers for the test set|
|`06\\\_pipeline\\\_testing`|Runs the evaluation and writes results to `tests/outputs/`|

Neo4j Browser: http://localhost:7474 (default credentials `neo4j` / `password`, local development only).

## Example

```
Q: What temperature did sensor 25.0 measure on 2004-02-28 at 00:58:46?
A: Temperature of sensor 25.0 at 2004-02-28 00:58:46: 19.616°C

Q: What is the maximum light value in epoch 957?
→ the organizer computes the maximum deterministically; the LLM only formats the answer.
```

## Project structure

```
src/graph/     GraphRAG components (query processor, retriever, organizer, generator, pipeline)
src/base/      TextRAG baseline (BM25)
notebooks/     Preprocessing, graph construction, pipeline tests, evaluation
tests/         Test set with gold labels, evaluation outputs
docker/        Docker Compose setup (Neo4j, Ollama, Jupyter)
```

## Author

David Wiese — [LinkedIn](https://www.linkedin.com/in/davidwiese)



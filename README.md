# LUXComplianceRAG

Luxembourg financial compliance RAG — ingest CSSF circulars and build a retrieval pipeline.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ingest (phase 1)

Downloads CSSF circular PDFs via the [publications RSS feed](https://www.cssf.lu/en/publications/):

```bash
python3 src/ingest.py
```

PDFs are saved under `data/` (gitignored). Logs go to `logs/ingest.log`.

## Parse & chunk (phase 2)

Extracts text from `data/*.pdf` (per-page metadata) and chunks with `langchain-text-splitters`:

```bash
python3 src/parser.py
```

Output: `data/chunks/chunks.jsonl`. Logs: `logs/parser.log`.

## Vector store (phase 3)

Embeds chunks with `paraphrase-multilingual-MiniLM-L12-v2` and stores them in local Qdrant:

```bash
python3 src/vector_store.py
```

Database: `data/qdrant_db/`. Logs: `logs/vector_store.log`. Re-running rebuilds the collection from scratch.

## Status

- [x] RSS discovery → document pages → PDF download
- [x] Text extraction (`pypdf`) + chunking (`langchain-text-splitters`)
- [x] Embeddings + vector store (Qdrant + sentence-transformers)
- [ ] Query / RAG answers

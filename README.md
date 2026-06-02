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

## Status

- [x] RSS discovery → document pages → PDF download
- [x] Text extraction (`pypdf`) + chunking (`langchain-text-splitters`)
- [ ] Embeddings + vector store
- [ ] Query / RAG answers

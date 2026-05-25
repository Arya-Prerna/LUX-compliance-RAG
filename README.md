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

## Status

- [x] RSS discovery → document pages → PDF download
- [ ] Text extraction (`pypdf`)
- [ ] Chunking, embeddings, vector store
- [ ] Query / RAG answers

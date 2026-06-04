import hashlib
import json
import logging
import os
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/vector_store.log"),
        logging.StreamHandler(),
    ],
)

CHUNKS_FILE = Path("data/chunks/chunks.jsonl")
QDRANT_PATH = "data/qdrant_db"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384
UPSERT_BATCH_SIZE = 100
ENCODE_BATCH_SIZE = 32


def _point_id(chunk_id: str) -> int:
    """Stable Qdrant point id from parser chunk id (e.g. cssf26_912.pdf:3)."""
    digest = hashlib.sha256(chunk_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


class LuxVectorStore:
    def __init__(self, collection_name: str = "lux_circulars"):
        self.collection_name = collection_name
        self.client = QdrantClient(path=QDRANT_PATH)

        logging.info("Loading embedding model: %s", MODEL_NAME)
        self.model = SentenceTransformer(MODEL_NAME)

    def _recreate_collection(self) -> None:
        """Drop and recreate so re-indexing does not leave stale points."""
        collections = self.client.get_collections().collections
        if any(c.name == self.collection_name for c in collections):
            logging.info("Deleting existing collection: %s", self.collection_name)
            self.client.delete_collection(self.collection_name)

        logging.info("Creating collection: %s", self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    def _load_chunks(self, chunks_path: Path) -> list[dict]:
        chunks: list[dict] = []
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    def _build_points(self, chunks: list[dict], vectors: list) -> list[PointStruct]:
        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            meta = chunk["metadata"]
            chunk_id = chunk["id"]
            points.append(
                PointStruct(
                    id=_point_id(chunk_id),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk_id,
                        "text": chunk["text"],
                        "source": meta["source"],
                        "page": meta["page"],
                        "chunk_index": meta["chunk_id"],
                    },
                )
            )
        return points

    def upsert_chunks(
        self,
        chunks_path: Path = CHUNKS_FILE,
        recreate_collection: bool = True,
    ) -> int:
        """Read JSONL chunks, embed in batches, upload to Qdrant."""
        chunks_path = Path(chunks_path)
        logging.info("Reading chunks from %s", chunks_path)
        chunks = self._load_chunks(chunks_path)
        if not chunks:
            logging.error("No chunks found in %s", chunks_path)
            return 0

        if recreate_collection:
            self._recreate_collection()

        texts = [c["text"] for c in chunks]
        logging.info("Embedding %d chunks (batch_size=%d)...", len(texts), ENCODE_BATCH_SIZE)
        all_vectors = self.model.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=True,
        )

        points = self._build_points(chunks, all_vectors)
        logging.info("Upserting %d points to Qdrant...", len(points))

        for start in tqdm(range(0, len(points), UPSERT_BATCH_SIZE), desc="Upserting"):
            batch = points[start : start + UPSERT_BATCH_SIZE]
            self.client.upsert(collection_name=self.collection_name, points=batch)

        logging.info("Vectorization complete. %d points in '%s'.", len(points), self.collection_name)
        return len(points)


if __name__ == "__main__":
    if CHUNKS_FILE.exists():
        store = LuxVectorStore()
        count = store.upsert_chunks(CHUNKS_FILE)
        if count:
            print(f"\nSuccess! Indexed {count} chunks in Qdrant ({QDRANT_PATH})")
    else:
        print(f"Error: {CHUNKS_FILE} not found. Run: python3 src/parser.py")

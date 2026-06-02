import json
import logging
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from tqdm import tqdm

if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/parser.log"),
        logging.StreamHandler(),
    ],
)

DATA_DIR = Path("data")
CHUNKS_FILE = DATA_DIR / "chunks" / "chunks.jsonl"


class CSSFParser:
    def __init__(self, input_dir="data", chunk_size=1000, chunk_overlap=200):
        self.input_dir = Path(input_dir)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def extract_pages(self, pdf_path: Path) -> list[dict] | None:
        """Extract text per page for citation metadata."""
        try:
            reader = PdfReader(str(pdf_path))
            pages: list[dict] = []
            for page_num, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append({"page": page_num, "text": text})
            return pages
        except Exception as e:
            logging.error(f"Error reading {pdf_path.name}: {e}")
            return None

    def chunk_pdf(self, pdf_path: Path) -> list[dict]:
        """Chunk one PDF page-by-page so metadata includes page numbers."""
        file_name = pdf_path.name
        pages = self.extract_pages(pdf_path)

        if pages is None:
            return []

        if not pages:
            logging.warning(f"No text extracted from {file_name}")
            return []

        chunks: list[dict] = []
        global_index = 0
        for page_data in pages:
            splits = self.splitter.split_text(page_data["text"])
            for split_text in splits:
                chunks.append(
                    {
                        "id": f"{file_name}:{global_index}",
                        "text": split_text,
                        "metadata": {
                            "source": file_name,
                            "page": page_data["page"],
                            "chunk_id": global_index,
                        },
                    }
                )
                global_index += 1
        return chunks

    def process_all_pdfs(self) -> list[dict]:
        pdf_files = sorted(self.input_dir.glob("*.pdf"))
        if not pdf_files:
            logging.error(
                f"No PDFs found in {self.input_dir.resolve()}. Run src/ingest.py first."
            )
            return []

        logging.info(f"Found {len(pdf_files)} PDF(s) to process.")
        all_chunks: list[dict] = []

        for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
            logging.info(f"Parsing {pdf_path.name}...")
            file_chunks = self.chunk_pdf(pdf_path)
            logging.info(f"  → {len(file_chunks)} chunk(s)")
            all_chunks.extend(file_chunks)

        logging.info(
            f"Created {len(all_chunks)} chunks from {len(pdf_files)} document(s)."
        )
        return all_chunks

    def save_chunks(self, chunks: list[dict], output_path: Path = CHUNKS_FILE) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        logging.info(f"Wrote {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    parser = CSSFParser()
    chunks = parser.process_all_pdfs()

    if chunks:
        parser.save_chunks(chunks)
        first = chunks[0]
        print(
            f"\nExample chunk (id={first['id']}, "
            f"source={first['metadata']['source']}, page={first['metadata']['page']}):"
        )
        print("-" * 50)
        print(first["text"][:500] + ("..." if len(first["text"]) > 500 else ""))
        print("-" * 50)
        print(f"\nSuccess! {len(chunks)} chunks saved to {CHUNKS_FILE}")
    else:
        print("No chunks produced. Check logs/parser.log")

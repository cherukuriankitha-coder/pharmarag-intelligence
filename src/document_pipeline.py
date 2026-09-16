"""Document ingestion and chunking utilities for PharmaRAG Intelligence."""
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    page: int
    source: str


def extract_pdf(path: str) -> List[str]:
    """Extract text page-by-page from a digital PDF."""
    reader = PdfReader(path)
    return [(page.extract_text() or "").strip() for page in reader.pages]


def chunk_pages(pages: List[str], source: str, chunk_size: int = 900, overlap: int = 150) -> List[Chunk]:
    """Create overlapping character chunks while retaining page metadata."""
    chunks: List[Chunk] = []
    step = max(1, chunk_size - overlap)
    for page_number, text in enumerate(pages, start=1):
        for start in range(0, len(text), step):
            piece = text[start:start + chunk_size].strip()
            if piece:
                chunks.append(Chunk(piece, page_number, Path(source).name))
    return chunks

"""Starter PharmaRAG pipeline for document question answering."""
from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    text: str
    page: int
    score: float = 0.0

def chunk_text(text: str, chunk_size: int = 180, overlap: int = 30) -> List[str]:
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks

def keyword_retrieve(query: str, chunks: List[Chunk], top_k: int = 3) -> List[Chunk]:
    terms = set(query.lower().split())
    for chunk in chunks:
        tokens = set(chunk.text.lower().split())
        chunk.score = len(terms & tokens) / max(len(terms), 1)
    return sorted(chunks, key=lambda item: item.score, reverse=True)[:top_k]

if __name__ == "__main__":
    sample = "Pharmaceutical documents contain storage conditions, supplier records, and packaging specifications."
    chunks = [Chunk(text=c, page=1) for c in chunk_text(sample)]
    for result in keyword_retrieve("storage conditions", chunks):
        print(result)

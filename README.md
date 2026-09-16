# PharmaRAG Intelligence

AI-powered document intelligence pipeline for retrieving evidence from pharmaceutical PDFs and supporting grounded question answering.

## Overview
The project demonstrates an end-to-end RAG workflow: PDF ingestion, page-aware chunking, semantic embeddings, similarity retrieval, and source metadata. It is structured as a portfolio implementation that can be extended with OCR, FAISS, an open-source generator, and a Gradio interface.

## Tech Stack
- Python
- PyPDF
- Sentence Transformers / MiniLM
- NumPy
- Retrieval-Augmented Generation (RAG)
- FAISS / Gradio (extension path)

## Architecture
`PDF -> Text Extraction -> Page-aware Chunks -> Embeddings -> Semantic Search -> Retrieved Evidence -> Answer Layer`

## Repository Structure
```text
src/
  document_pipeline.py   # PDF extraction and overlapping chunking
  retriever.py           # semantic embedding and top-k retrieval
```

## Key Features
- Preserves page and source metadata during chunking
- Uses normalized sentence embeddings for semantic similarity
- Returns ranked chunks with relevance scores
- Modular design separates ingestion from retrieval

## Example
```python
from src.document_pipeline import extract_pdf, chunk_pages
from src.retriever import SemanticRetriever

pages = extract_pdf("sample.pdf")
chunks = chunk_pages(pages, "sample.pdf")
retriever = SemanticRetriever()
retriever.fit(chunks)
results = retriever.search("What are the recommended storage conditions?")
```

## Roadmap
- OCR fallback for scanned pages
- FAISS persistence
- metadata filtering
- answer generation with citations
- Gradio demo
- retrieval evaluation using Recall@K and MRR

## Portfolio Note
This repository is an evolving implementation. Reported performance metrics should only be added after running a documented evaluation dataset.
"""Build an embeddings index (FAISS) from a PDF.

Usage:
  python build_index.py --pdf Sankirtan\ Madhuri.pdf

Output files:
  index.faiss        - FAISS index
  index_meta.npy     - numpy array of dicts with fields: page, text
  embed_model.pkl    - model name saved for reference
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import fitz
from sentence_transformers import SentenceTransformer
import faiss


def extract_page_texts(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    pages = []
    for p in doc:
        pages.append(p.get_text())
    return pages


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index(pdf_path: str, model_name: str = "all-MiniLM-L6-v2") -> None:
    pages = extract_page_texts(pdf_path)
    items = []
    for i, ptext in enumerate(pages):
        chunks = chunk_text(ptext)
        for c in chunks:
            items.append({"page": i, "text": c})

    texts = [it["text"] for it in items]
    print(f"Computing embeddings for {len(texts)} chunks using {model_name}...")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # normalize embeddings for cosine similarity with inner product
    faiss.normalize_L2(emb)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    idx_path = f"{base}.index.faiss"
    meta_path = f"{base}.index_meta.npy"
    model_ref = f"{base}.embed_model.json"
    faiss.write_index(index, idx_path)
    np.save(meta_path, np.array(items, dtype=object))
    with open(model_ref, "w", encoding="utf-8") as f:
        json.dump({"model": model_name}, f)
    print("Index built:", idx_path, meta_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True)
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    return p.parse_args()


def main():
    args = parse_args()
    build_index(args.pdf, args.model)


if __name__ == '__main__':
    main()

from __future__ import annotations
import argparse
import contextlib
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import fitz
from sentence_transformers import SentenceTransformer
import faiss


@contextlib.contextmanager
def _suppress_stderr():
    old = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 2)
    os.close(null)
    try:
        yield
    finally:
        os.dup2(old, 2)
        os.close(old)


def extract_page_texts(pdf_path: str, use_ocr: bool = False) -> list[str]:
    doc = fitz.open(pdf_path)
    pages = []
    if use_ocr:
        import torch
        import os
        from PIL import Image
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        from surya.common.surya.schema import TaskNames
        device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["TORCH_DEVICE"] = device
        print(f"Loading Surya OCR models on {device}...")
        foundation_predictor = FoundationPredictor()
        rec_predictor = RecognitionPredictor(foundation_predictor)
        det_predictor = DetectionPredictor()
        mat = fitz.Matrix(2, 2)
        total = len(doc)
        for i, p in enumerate(doc):
            pix = p.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            result = rec_predictor([img], [TaskNames.ocr_with_boxes], det_predictor)[0]
            text = "\n".join(line.text for line in result.text_lines if line.text.strip())
            pages.append(text)
            if (i + 1) % 10 == 0:
                print(f"  OCR: {i+1}/{total} pages done", flush=True)
    else:
        for p in doc:
            pages.append(p.get_text())
    return pages


def _is_mostly_devanagari(text: str, threshold: float = 0.4) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    devanagari = sum(1 for c in letters if 'ऀ' <= c <= 'ॿ')
    return devanagari / len(letters) >= threshold


def _clean_line(text: str) -> str:
    """Strip garbage ASCII symbols that PyMuPDF picks up from malformed PDF encoding."""
    cleaned = re.sub(r'[^ऀ-ॿ\sa-zA-Z0-9।,\.\-\!\?]', '', text)
    return re.sub(r' {2,}', ' ', cleaned).strip()


def chunk_lines(text: str, min_chars: int = 5, max_chars: int = 200, devanagari_only: bool = False) -> list[str]:
    lines = []
    for line in text.splitlines():
        line = _clean_line(line.strip())
        if not (min_chars <= len(line) <= max_chars):
            continue
        if devanagari_only and not _is_mostly_devanagari(line):
            continue
        lines.append(line)
    return lines


def detect_bhajan_starts(pages: list[str]) -> list[int]:
    """Return page indices where a new bhajan begins.

    Heuristic: a page starts a new bhajan when its first non-empty line is
    noticeably shorter than the page's average line length (title pattern),
    or the page has very few lines (separator / title-only page).
    """
    starts = [0]
    for i, page in enumerate(pages[1:], 1):
        lines = [l.strip() for l in page.splitlines() if l.strip()]
        if not lines:
            continue
        if len(lines) <= 2:
            starts.append(i)
            continue
        first = lines[0]
        rest_avg = sum(len(l) for l in lines[1:]) / len(lines[1:])
        if len(first) <= 60 and rest_avg > 0 and len(first) < rest_avg * 0.65:
            starts.append(i)
    return sorted(set(starts))


def extract_bhajan_titles(pages: list[str], bhajan_starts: list[int]) -> list[str]:
    """Return a display title for each bhajan (first Devanagari-heavy line at start page)."""
    titles = []
    for start_page in bhajan_starts:
        lines = [l.strip() for l in pages[start_page].splitlines()
                 if l.strip() and len(l.strip()) >= 3]
        devanagari_lines = [l for l in lines if _is_mostly_devanagari(l)]
        line = devanagari_lines[0] if devanagari_lines else (lines[0] if lines else "")
        titles.append(line[:80] or f"Bhajan {len(titles) + 1}")
    return titles


def build_tfidf_weights(chunks: list[dict]) -> dict[str, float]:
    """Compute IDF weights across bhajans.

    High IDF → discriminative word (appears in few bhajans, e.g. a unique name).
    Low/zero IDF → common word (appears in every bhajan, e.g. "Radha").
    """
    bhajan_texts: dict[int, str] = defaultdict(str)
    for c in chunks:
        bhajan_texts[c.get("bhajan_id", 0)] += " " + c["text"]

    n = len(bhajan_texts)
    if n < 2:
        return {}

    df: dict[str, int] = defaultdict(int)
    for text in bhajan_texts.values():
        words = set(re.findall(r'[\wऀ-ॿ]+', text.lower()))
        for w in words:
            df[w] += 1

    return {
        w: math.log(n / cnt)
        for w, cnt in df.items()
        if cnt < n  # words in every bhajan get IDF=0, skip them
    }


def build_index(pdf_path: str, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", use_ocr: bool = False) -> None:
    pages = extract_page_texts(pdf_path, use_ocr=use_ocr)

    # Detect bhajan boundaries and stamp each chunk with a bhajan_id
    bhajan_starts = detect_bhajan_starts(pages)
    page_to_bhajan: dict[int, int] = {}
    for bid, start in enumerate(bhajan_starts):
        end = bhajan_starts[bid + 1] if bid + 1 < len(bhajan_starts) else len(pages)
        for p in range(start, end):
            page_to_bhajan[p] = bid
    print(f"Detected {len(bhajan_starts)} bhajan(s) starting at pages: {bhajan_starts}")

    items = []
    for i, ptext in enumerate(pages):
        for line in chunk_lines(ptext, devanagari_only=use_ocr):
            items.append({"page": i, "text": line, "bhajan_id": page_to_bhajan.get(i, 0)})

    tfidf = build_tfidf_weights(items)
    titles = extract_bhajan_titles(pages, bhajan_starts)

    texts = [it["text"] for it in items]
    print(f"Computing embeddings for {len(texts)} chunks using {model_name}...")
    with _suppress_stderr():
        model = SentenceTransformer(model_name, device="cuda")
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=256)

    # normalize embeddings for cosine similarity with inner product
    faiss.normalize_L2(emb)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    idx_path = f"{base}.index.faiss"
    meta_path = f"{base}.index_meta.npy"
    model_ref = f"{base}.embed_model.json"
    tfidf_path = f"{base}.tfidf.json"
    bhajans_path = f"{base}.bhajans.json"
    faiss.write_index(index, idx_path)
    np.save(meta_path, np.array(items, dtype=object))
    with open(model_ref, "w", encoding="utf-8") as f:
        json.dump({"model": model_name}, f)
    with open(tfidf_path, "w", encoding="utf-8") as f:
        json.dump(tfidf, f, ensure_ascii=False)
    bhajans_data = [
        {"bhajan_id": bid, "title": titles[bid], "start_page": bhajan_starts[bid]}
        for bid in range(len(bhajan_starts))
    ]
    with open(bhajans_path, "w", encoding="utf-8") as f:
        json.dump(bhajans_data, f, ensure_ascii=False, indent=2)
    print(
        f"Index built: {len(items)} chunks, {len(bhajan_starts)} bhajan(s), "
        f"{len(tfidf)} discriminative words"
    )
    print("Files:", idx_path, meta_path, model_ref, tfidf_path, bhajans_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True)
    p.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--ocr", action="store_true", help="Use EasyOCR to extract Hindi text instead of PyMuPDF")
    return p.parse_args()


def main():
    args = parse_args()
    build_index(args.pdf, args.model, use_ocr=args.ocr)


if __name__ == '__main__':
    main()

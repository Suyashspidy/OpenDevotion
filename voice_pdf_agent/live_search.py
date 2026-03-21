from __future__ import annotations
import argparse
import threading
import tempfile
import os
import time

import sounddevice as sd
import soundfile as sf
import whisper
import re
import fitz
from PIL import Image, ImageTk
import tkinter as tk
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from pathlib import Path


def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    images = []
    for p in doc:
        text = p.get_text()
        pix = p.get_pixmap(alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(text)
        images.append(img)
    return pages, images


class PDFViewer(tk.Tk):
    def __init__(self, images, pages_text):
        super().__init__()
        self.title("Live PDF Search")
        self.canvas = tk.Canvas(self)
        self.vbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.frame, anchor='nw')
        self.images = images
        self.pages_text = pages_text
        self.photo_refs = []
        self.page_positions = []
        self.labels = []
        self._populate()
        self.frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def _populate(self):
        y = 0
        for i, img in enumerate(self.images):
            max_w = 900
            w, h = img.size
            if w > max_w:
                ratio = max_w / w
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
                w, h = img.size
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(self.frame, image=photo)
            lbl.image = photo
            lbl.pack(padx=4, pady=6)
            self.photo_refs.append(photo)
            self.labels.append(lbl)
            self.page_positions.append(y)
            y += h + 12

    def highlight_and_scroll(self, page_index, query):
        if page_index < 0 or page_index >= len(self.images):
            return
        target_y = self.page_positions[page_index]
        total = max(1, self.canvas.bbox("all")[3])
        fraction = target_y / total
        self.canvas.yview_moveto(fraction)
        # highlight the label for the matched page
        try:
            lbl = self.labels[page_index]
        except Exception:
            return
        prev_bg = lbl.cget('bg')
        prev_bd = lbl.cget('highlightthickness') if 'highlightthickness' in lbl.keys() else 0
        lbl.config(bg='yellow')
        def clear_label():
            try:
                lbl.config(bg=prev_bg)
            except Exception:
                pass
        self.after(2000, clear_label)


def audio_loop(model, viewer, pages_text, samplerate=16000, chunk_s=3, embed_model=None, faiss_index=None, index_meta=None):
    print("Starting audio loop (speak now)...")
    while True:
        try:
            print("Recording...")
            recording = sd.rec(int(chunk_s * samplerate), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()
            audio = recording.flatten()
            # Transcribe directly from numpy array to avoid ffmpeg subprocess on Windows
            result = model.transcribe(audio, language='en', fp16=False)
            txt = result.get('text', '').strip()
            # sanitize common polite closings that may be appended by background noise
            def sanitize_transcript(s: str) -> str:
                if not s:
                    return s
                s = s.strip()
                # remove trailing polite phrases like 'thank you', 'thanks', etc.
                s = re.sub(r"(?i)\b(?:thankyou|thank you(?: very much)?|thanks(?: a lot)?)\b[\s\.,!]*$", "", s).strip()
                return s

            txt = sanitize_transcript(txt)
            if not txt:
                continue
            print("Heard:", txt)
            # If FAISS index available, use semantic retrieval
            best_page = -1
            display_text = txt
            if faiss_index is not None and embed_model is not None and index_meta is not None:
                try:
                    q_emb = embed_model.encode([txt], convert_to_numpy=True)
                    faiss.normalize_L2(q_emb)
                    D, I = faiss_index.search(q_emb, 3)
                    # I is shape (1, k)
                    top_idx = int(I[0][0])
                    if top_idx >= 0:
                        meta = index_meta[top_idx].item()
                        best_page = int(meta.get('page', -1))
                        # choose the matched substring from that page text
                        matched = find_best_match_in_page(pages_text[best_page], txt)
                        if matched:
                            display_text = matched
                except Exception as e:
                    print('Retrieval error:', e)

            # fallback to fuzzy page search if retrieval didn't find page
            if best_page < 0:
                best_score = 0
                for i, ptext in enumerate(pages_text):
                    score = fuzz.partial_ratio(txt.lower(), ptext.lower())
                    if score > best_score:
                        best_score = score
                        best_page = i
                print(f"Best page {best_page} (score={best_score})")
                if best_score > 30:
                    matched = find_best_match_in_page(pages_text[best_page], txt)
                    if matched:
                        display_text = matched

            if best_page >= 0:
                viewer.after(0, viewer.highlight_and_scroll, best_page, display_text)
        except Exception as e:
            print("Audio loop error:", e)
            time.sleep(1)


def find_best_match_in_page(page_text: str, query: str) -> str | None:
    """Return a short substring from page_text that best matches query.

    Strategy:
    - Try exact case-insensitive substring of the full query.
    - Split page into candidate sentences and use rapidfuzz to pick best sentence.
    - Within best sentence, try to find the longest n-gram from query that appears exactly.
    - If exact n-gram not found, return the best sentence trimmed.
    """
    if not page_text or not query:
        return None
    q = query.strip()
    # exact substring
    idx = page_text.lower().find(q.lower())
    if idx != -1:
        return page_text[idx: idx + len(q)].strip()

    # split into sentences (simple heuristic)
    candidates = re.split(r"(?<=[\.!?\n])\s+", page_text)
    # choose best sentence by partial_ratio
    choice = None
    best = 0
    for s in candidates:
        if not s.strip():
            continue
        score = fuzz.partial_ratio(q.lower(), s.lower())
        if score > best:
            best = score
            choice = s

    if not choice:
        return None

    # try to find longest exact n-gram from query inside the chosen sentence
    words = [w for w in re.findall(r"\w+", q)]
    for n in range(len(words), 0, -1):
        for i in range(0, len(words) - n + 1):
            ngram = " ".join(words[i:i + n])
            if ngram and ngram.lower() in choice.lower():
                # return the exact substring from choice (preserve original casing)
                start = choice.lower().find(ngram.lower())
                if start != -1:
                    return choice[start:start + len(ngram)].strip()

    # fallback: return trimmed best sentence (limit length)
    trimmed = choice.strip()
    if len(trimmed) > 120:
        trimmed = trimmed[:117].rsplit(' ', 1)[0] + '...'
    return trimmed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pdf', required=True)
    p.add_argument('--test', action='store_true', help='Run extraction and model load only (no GUI/mic)')
    p.add_argument('--use_index', action='store_true', help='Use FAISS index for retrieval if available')
    p.add_argument('--build_index', action='store_true', help='Build FAISS index from PDF on start')
    args = p.parse_args()
    pdf_path = args.pdf
    print('Loading PDF...')
    pages_text, images = extract_pages(pdf_path)
    print(f'Extracted {len(pages_text)} pages')
    if args.test:
        print('Test mode: loading whisper model only...')
        model = whisper.load_model('small')
        print('Model loaded successfully')
        return
    model = whisper.load_model('small')
    # prepare retrieval index if requested
    embed_model = None
    faiss_index = None
    index_meta = None
    base = Path(pdf_path).stem
    idx_path = f"{base}.index.faiss"
    meta_path = f"{base}.index_meta.npy"
    model_ref = f"{base}.embed_model.json"
    if args.build_index:
        from build_index import build_index
        build_index(pdf_path)
    if args.use_index:
        try:
            if Path(idx_path).exists() and Path(meta_path).exists():
                faiss_index = faiss.read_index(idx_path)
                index_meta = np.load(meta_path, allow_pickle=True)
                with open(model_ref, 'r', encoding='utf-8') as f:
                    mm = json.load(f)
                    embed_model = SentenceTransformer(mm.get('model', 'all-MiniLM-L6-v2'))
                print('Loaded FAISS index and embedding model')
            else:
                print('Index files not found; run with --build_index or run build_index.py')
        except Exception as e:
            print('Failed to load index:', e)

    viewer = PDFViewer(images, pages_text)
    t = threading.Thread(target=audio_loop, args=(model, viewer, pages_text, 16000, 3, embed_model, faiss_index, index_meta), daemon=True)
    t.start()
    viewer.mainloop()


if __name__ == '__main__':
    main()

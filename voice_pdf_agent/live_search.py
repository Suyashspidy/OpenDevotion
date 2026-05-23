from __future__ import annotations
import argparse
import contextlib
import math
import os
import threading
import time
from collections import deque

import re
import fitz
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from rapidfuzz import fuzz
import numpy as np
import json
from pathlib import Path

# Suppress transformers INFO/WARNING noise (LOAD REPORT, logits-processor duplicates, etc.)
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
except Exception:
    pass

# Suppress huggingface_hub unauthenticated-request warning (models are local/cached)
try:
    from huggingface_hub.utils import logging as _hub_logging
    _hub_logging.set_verbosity_error()
except Exception:
    pass


@contextlib.contextmanager
def _suppress_stderr():
    """Redirect fd-level stderr to null during model loading.

    safetensors 0.7+ prints a 'LOAD REPORT' directly via Rust eprintln!()
    to file-descriptor 2, bypassing Python's sys.stderr — only an OS-level
    dup2 redirect can suppress it.

    On Windows, os.dup2 can raise OSError(WinError 1) when called from a
    non-main daemon thread. In that case we skip suppression and yield normally.
    """
    try:
        old = os.dup(2)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 2)
        os.close(null)
    except OSError:
        yield
        return
    try:
        yield
    finally:
        try:
            os.dup2(old, 2)
            os.close(old)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Voice Activity Detection — silero-vad with energy fallback
# ---------------------------------------------------------------------------

_vad_state: tuple | None = None  # (model, get_speech_timestamps) once loaded


def _vad_load():
    """Lazy-load silero-vad from torch.hub (cached after first download)."""
    global _vad_state
    if _vad_state is not None:
        return _vad_state
    import torch
    model, utils = torch.hub.load(
        'snakers4/silero-vad', 'silero_vad',
        force_reload=False, verbose=False,
    )
    _vad_state = (model, utils[0])  # utils[0] = get_speech_timestamps
    print("[VAD] silero-vad loaded.")
    return _vad_state


def vad_has_speech(
    audio: np.ndarray,
    samplerate: int = 16000,
    threshold: float = 0.5,
) -> bool:
    """Return True if silero-vad finds at least one speech segment.

    threshold=0.5 is conservative enough to reject tabla/harmonium bleed
    while still catching soft singing. Falls back to an RMS energy check
    if the model isn't available.
    """
    try:
        model, get_ts = _vad_load()
        import torch
        t = torch.from_numpy(audio).float()
        segments = get_ts(t, model, sampling_rate=samplerate, threshold=threshold)
        return len(segments) > 0
    except Exception as e:
        print(f"[VAD] silero unavailable ({e}), using energy fallback.")
        return float(np.sqrt(np.mean(audio ** 2))) > 0.008


def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    images = []
    mat = fitz.Matrix(2, 2)  # 2x zoom → ~144 DPI, crisp on screen
    for p in doc:
        text = p.get_text()
        pix = p.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(text)
        images.append(img)
    return pages, images, doc


class PDFViewer(tk.Toplevel):
    MAX_W = 900

    def __init__(self, images, pages_text, doc=None, master=None, position_tracker=None):
        if master is None:
            self._tk_root = tk.Tk()
            self._tk_root.withdraw()
            master = self._tk_root
        else:
            self._tk_root = None
        super().__init__(master)
        self.title("Live PDF Search")
        self.position_tracker = position_tracker
        self.images = images
        self.pages_text = pages_text
        self.doc = doc
        self._current_page = 0
        self._photo_ref = None  # single live PhotoImage reference

        toolbar = tk.Frame(self, bg='#1e1e1e', pady=3)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self._prev_btn = tk.Button(
            toolbar, text="◀", command=self._prev_page,
            bg='#3a3a3a', fg='white', relief=tk.FLAT,
            font=('Helvetica', 9, 'bold'), padx=8, pady=2,
            cursor='hand2',
        )
        self._prev_btn.pack(side=tk.LEFT, padx=(8, 2))

        self._page_label = tk.Label(
            toolbar, text="Page —", bg='#1e1e1e', fg='#999999',
            font=('Helvetica', 9),
        )
        self._page_label.pack(side=tk.LEFT, padx=6)

        self._next_btn = tk.Button(
            toolbar, text="▶", command=self._next_page,
            bg='#3a3a3a', fg='white', relief=tk.FLAT,
            font=('Helvetica', 9, 'bold'), padx=8, pady=2,
            cursor='hand2',
        )
        self._next_btn.pack(side=tk.LEFT, padx=(2, 8))

        if position_tracker is not None:
            self._lock_btn = tk.Button(
                toolbar, text="📍  Lock Position Here",
                command=self._lock_position,
                bg='#2d7a2d', fg='white', relief=tk.FLAT,
                font=('Helvetica', 9, 'bold'), padx=10, pady=2,
                cursor='hand2',
            )
            self._lock_btn.pack(side=tk.RIGHT, padx=8)

        self._img_label = tk.Label(self, bg='#2b2b2b')
        self._img_label.pack(fill=tk.BOTH, expand=True)

        self.bind('<Left>', lambda e: self._prev_page())
        self.bind('<Right>', lambda e: self._next_page())

        # Manual-override: when the operator scrolls, pause auto-scroll for
        # MANUAL_OVERRIDE_S seconds so they can show the audience the correct
        # lyrics without the model immediately jumping away.
        self._manual_until: float = 0.0
        self._MANUAL_OVERRIDE_S: float = 8.0

        self._show_page(0)

    def _scale_image(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w > self.MAX_W:
            ratio = self.MAX_W / w
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return img

    def _show_page(self, page_index: int, photo=None) -> None:
        if page_index < 0 or page_index >= len(self.images):
            return
        self._current_page = page_index
        self._page_label.config(text=f"Page {page_index + 1} / {len(self.images)}")
        if photo is None:
            photo = ImageTk.PhotoImage(self._scale_image(self.images[page_index]))
        self._photo_ref = photo
        self._img_label.config(image=photo)
        self._img_label.image = photo

    def _prev_page(self) -> None:
        self._show_page(self._current_page - 1)
        self._on_manual_scroll()

    def _next_page(self) -> None:
        self._show_page(self._current_page + 1)
        self._on_manual_scroll()

    def _on_manual_scroll(self) -> None:
        """Pause auto-scroll and sync tracker to the page the operator chose."""
        self._manual_until = time.time() + self._MANUAL_OVERRIDE_S
        if self.position_tracker is not None:
            self.position_tracker.lock_to_page(self._current_page)

    def _lock_position(self) -> None:
        page = self._current_page
        self.position_tracker.lock_to_page(page)
        self._lock_btn.config(text=f"📍  Locked — page {page + 1}", bg='#1a5c1a')
        self.after(3000, lambda: self._lock_btn.config(
            text="📍  Lock Position Here", bg='#2d7a2d'))

    def scroll_to_page(self, page_index: int) -> None:
        try:
            self._show_page(page_index)
        except Exception:
            pass

    def highlight_and_scroll(self, page_index: int, query: str) -> None:
        try:
            if page_index < 0 or page_index >= len(self.images):
                return
            highlighted = self._make_highlighted_photo(page_index, query)
            self._show_page(page_index, photo=highlighted)
            if highlighted:
                def restore():
                    if self._current_page == page_index:
                        self._show_page(page_index)
                self.after(3000, restore)
        except Exception:
            pass

    def _make_highlighted_photo(self, page_index: int, query: str):
        if not self.doc or not query:
            return None
        try:
            page = self.doc[page_index]
            rects = page.search_for(query)
            if not rects:
                rects = []
                for word in query.split():
                    if len(word) > 1:
                        rects.extend(page.search_for(word))
            if not rects:
                return None
            orig = self.images[page_index]
            img_rgba = orig.convert('RGBA')
            overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for r in rects:
                draw.rectangle([r.x0, r.y0, r.x1, r.y1], fill=(255, 220, 0, 180))
            composited = Image.alpha_composite(img_rgba, overlay).convert('RGB')
            scaled = self._scale_image(composited)
            return ImageTk.PhotoImage(scaled)
        except Exception as e:
            print("Lyric highlight error:", e)
            return None

    def mainloop(self, n=0):
        if self._tk_root is not None:
            self._tk_root.mainloop(n)
        else:
            super().mainloop(n)


def build_vocab_prompt(pages_text: list[str], max_chars: int = 800) -> str:
    """Build a deduplicated vocabulary string from the PDF to bias Whisper toward prayer-specific words."""
    all_text = " ".join(pages_text)
    words = re.findall(r"[ऀ-ॿA-Za-z']+", all_text)
    seen: set[str] = set()
    unique_words: list[str] = []
    for w in words:
        lw = w.lower()
        if lw not in seen and len(lw) > 2:
            seen.add(lw)
            unique_words.append(w)
    return ", ".join(unique_words)[:max_chars]


# ---------------------------------------------------------------------------
# IndicWhisper (HuggingFace) helpers
# ---------------------------------------------------------------------------

def load_indicwhisper(model_name_or_path: str):
    """Load IndicWhisper processor + model from HuggingFace or a local fine-tuned checkpoint."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    print(f"Loading IndicWhisper from '{model_name_or_path}'...")
    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    print(f"IndicWhisper loaded on {device}")
    return model, processor


def transcribe_indicwhisper(
    audio: np.ndarray,
    iw_model,
    iw_processor,
    language: str = "hi",
    samplerate: int = 16000,
) -> str:
    """Transcribe a raw float32 numpy array using IndicWhisper."""
    import torch
    device = next(iw_model.parameters()).device
    inputs = iw_processor(audio, sampling_rate=samplerate, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    # All-ones encoder attention mask — single clip, no padding
    attention_mask = torch.ones(
        input_features.shape[0], input_features.shape[-1],
        dtype=torch.long, device=device,
    )
    # Use forced_decoder_ids so the language/task tokens are injected once,
    # avoiding the duplicate SuppressTokensLogitsProcessor warning from generate()
    forced_decoder_ids = iw_processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )
    with torch.no_grad():
        predicted_ids = iw_model.generate(
            input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids,
            num_beams=5,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
    texts = iw_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return texts[0].strip() if texts else ""


# ---------------------------------------------------------------------------
# Chunk text cleaning (used by BhajanTracker scoring AND display)
# ---------------------------------------------------------------------------

def _clean_chunk_text(text: str) -> str:
    """Strip garbage PDF-extraction symbols.

    Keeps Devanagari, Latin letters, spaces, basic punctuation.
    Returns empty string if result is < 40% Devanagari — callers treat
    that as a garbage chunk and skip scoring / display.
    """
    cleaned = re.sub(r'[^ऀ-ॿ\sa-zA-Z।,\.\-\!\?]', '', text)
    cleaned = re.sub(r' {2,}', ' ', cleaned).strip()
    letters = [c for c in cleaned if c.isalpha()]
    if not letters:
        return ""
    deva = sum(1 for c in letters if 'ऀ' <= c <= 'ॿ')
    if deva / len(letters) < 0.4:
        return ""
    return cleaned


# ---------------------------------------------------------------------------
# Hybrid retrieval: FAISS semantic + fuzzy keyword combined simultaneously
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    pages_text: list[str],
    embed_model=None,
    faiss_index=None,
    index_meta=None,
    semantic_weight: float = 0.6,
) -> tuple[int, float]:
    """Return (best_page, combined_score) by fusing semantic and keyword scores.

    Both retrievers always run; their normalised scores are merged with a
    weighted sum so neither is merely a fallback for the other.
    semantic_weight controls the dense-vs-sparse trade-off (0 = fuzzy only,
    1 = semantic only).
    """
    n = len(pages_text)
    semantic_scores = np.zeros(n)
    fuzzy_scores = np.zeros(n)

    # --- dense / semantic (FAISS) -----------------------------------------
    has_index = faiss_index is not None and embed_model is not None and index_meta is not None
    if has_index:
        try:
            import faiss as _faiss
            q_emb = embed_model.encode([query], convert_to_numpy=True)
            _faiss.normalize_L2(q_emb)
            k = min(10, faiss_index.ntotal)
            D, I = faiss_index.search(q_emb, k)
            for dist, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                meta = index_meta[idx]
                if hasattr(meta, 'item'):
                    meta = meta.item()
                page = int(meta.get('page', -1))
                if 0 <= page < n:
                    # inner-product on L2-normalised vectors == cosine similarity in [-1,1]
                    semantic_scores[page] = max(semantic_scores[page], float(dist))
            # shift to [0, 1]
            semantic_scores = (semantic_scores + 1.0) / 2.0
        except Exception as e:
            print('Semantic retrieval error:', e)
            has_index = False

    # --- sparse / fuzzy keyword -------------------------------------------
    for i, ptext in enumerate(pages_text):
        fuzzy_scores[i] = fuzz.partial_ratio(query.lower(), ptext.lower()) / 100.0

    # --- score fusion -------------------------------------------------------
    if has_index:
        combined = semantic_weight * semantic_scores + (1.0 - semantic_weight) * fuzzy_scores
    else:
        combined = fuzzy_scores

    best_page = int(np.argmax(combined))
    return best_page, float(combined[best_page])


# ---------------------------------------------------------------------------
# Position Tracker — stateful agent with window search + Gemini recovery
# ---------------------------------------------------------------------------

class PositionTracker:
    WINDOW_FORWARD = 20
    WINDOW_BACKWARD = 5
    HIGH_CONF = 0.45
    LOW_CONF = 0.25
    STUCK_LIMIT = 3

    def __init__(self, chunks, embed_model=None, faiss_index=None, gemini_key=""):
        self.chunks = chunks          # list of {"page": int, "text": str}
        self.embed_model = embed_model
        self.faiss_index = faiss_index
        self.gemini_key = gemini_key
        self.current_idx = 0
        self.stuck_count = 0
        self.history: list[str] = []

    def _window_search(self, query: str) -> tuple[int, float]:
        lo = max(0, self.current_idx - self.WINDOW_BACKWARD)
        hi = min(len(self.chunks), self.current_idx + self.WINDOW_FORWARD)
        best_idx, best_score = self.current_idx, 0.0
        for i in range(lo, hi):
            score = fuzz.partial_ratio(query.lower(), self.chunks[i]["text"].lower()) / 100.0
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx, best_score

    def _global_search(self, query: str) -> tuple[int, float]:
        n = len(self.chunks)
        sem_scores = np.zeros(n)
        if self.faiss_index is not None and self.embed_model is not None:
            try:
                import faiss as _faiss
                q_emb = self.embed_model.encode([query], convert_to_numpy=True)
                _faiss.normalize_L2(q_emb)
                k = min(20, self.faiss_index.ntotal)
                D, I = self.faiss_index.search(q_emb, k)
                for dist, idx in zip(D[0], I[0]):
                    if 0 <= idx < n:
                        sem_scores[idx] = max(sem_scores[idx], (float(dist) + 1.0) / 2.0)
            except Exception as e:
                print("FAISS error:", e)
        fuzzy_scores = np.array([
            fuzz.partial_ratio(query.lower(), c["text"].lower()) / 100.0
            for c in self.chunks
        ])
        combined = 0.6 * sem_scores + 0.4 * fuzzy_scores
        best_idx = int(np.argmax(combined))
        return best_idx, float(combined[best_idx])

    def _gemini_recover(self) -> int:
        if not self.gemini_key or not self.history:
            return self.current_idx
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            step = max(1, len(self.chunks) // 25)
            landmarks = "\n".join(
                f"[{i}] {self.chunks[i]['text'][:80]}"
                for i in range(0, len(self.chunks), step)
            )
            recent = "\n".join(f"- {h}" for h in self.history[-5:])
            prompt = (
                "You are syncing live bhajan singing to a PDF lyric sheet.\n"
                f"Recent sung lines:\n{recent}\n\n"
                f"PDF landmarks (index: text):\n{landmarks}\n\n"
                "Which landmark index is closest to where the singer is? "
                "Reply with ONLY the integer index."
            )
            resp = model.generate_content(prompt)
            return int(resp.text.strip())
        except Exception as e:
            print("Gemini recovery error:", e)
            return self.current_idx

    def process(self, transcription: str) -> tuple[int, str]:
        self.history.append(transcription)
        if len(self.history) > 10:
            self.history.pop(0)

        win_idx, win_score = self._window_search(transcription)
        if win_score >= self.HIGH_CONF:
            self.current_idx = win_idx
            self.stuck_count = 0
            c = self.chunks[win_idx]
            return c["page"], c["text"]

        glob_idx, glob_score = self._global_search(transcription)
        if glob_score >= self.LOW_CONF:
            self.current_idx = glob_idx
            self.stuck_count = 0
            c = self.chunks[glob_idx]
            return c["page"], c["text"]

        self.stuck_count += 1
        print(f"[Tracker] Low confidence win={win_score:.2f} glob={glob_score:.2f} stuck={self.stuck_count}")
        if self.stuck_count >= self.STUCK_LIMIT:
            print("[Tracker] Calling Gemini for position recovery...")
            recovered = self._gemini_recover()
            self.current_idx = max(0, min(recovered, len(self.chunks) - 1))
            self.stuck_count = 0
            print(f"[Tracker] Reset to chunk {self.current_idx}")

        c = self.chunks[self.current_idx]
        return c["page"], c["text"]


# ---------------------------------------------------------------------------
# Bhajan-level state tracker (wraps position tracking with bhajan identity)
# ---------------------------------------------------------------------------

class BhajanTracker:
    """Tracks which bhajan is currently being sung and resists false jumps.

    Three mechanisms work together:
    1. Bhajan-level state — requires SWITCH_VOTES consecutive chunks all
       agreeing on the same new bhajan before switching. A single shared
       word like "Radha" cannot trigger a switch.
    2. TF-IDF discriminative boosting — words unique to few bhajans raise
       the score of the correct bhajan; ubiquitous words are down-weighted.
    3. Window search — within the current bhajan, position advances through
       a sliding forward window instead of searching globally every chunk.
    """

    SWITCH_VOTES     = 3
    HIGH_CONF        = 0.43   # confident window hit → stay, advance position
    LOW_CONF         = 0.25   # weak global hit → stay in current bhajan
    FREE_SEARCH_CONF = 0.20   # minimum score to accept a free-search escape (B)
    STUCK_ROUNDS     = 2      # consecutive low-conf rounds before B/D trigger
    WINDOW_FORWARD   = 20
    WINDOW_BACKWARD  = 5

    @property
    def context_window_size(self) -> int:
        """C: expand rolling context when stuck so the matcher sees more text."""
        return 7 if self._low_conf_streak >= self.STUCK_ROUNDS else 3

    def __init__(
        self,
        chunks: list[dict],
        embed_model=None,
        faiss_index=None,
        tfidf_weights: dict | None = None,
        gemini_key: str = "",
        setlist: list[int] | None = None,
        setlist_ordered: bool = True,
    ):
        self.chunks = chunks
        self.embed_model = embed_model
        self.faiss_index = faiss_index
        self.tfidf_weights = tfidf_weights or {}
        self.gemini_key = gemini_key

        # Group chunk indices by bhajan_id (falls back to 0 for old indexes)
        self.bhajan_ids: list[int] = sorted(
            set(c.get("bhajan_id", 0) for c in chunks)
        )
        self.bhajan_chunk_indices: dict[int, list[int]] = {
            bid: [] for bid in self.bhajan_ids
        }
        for i, c in enumerate(chunks):
            self.bhajan_chunk_indices[c.get("bhajan_id", 0)].append(i)

        # Pre-build per-bhajan text using only clean (non-garbage) chunks.
        # Garbage chunks score ~0.33 against any Hindi input (false floor) —
        # filtering them prevents artificially stable confidence values.
        self._bhajan_texts: dict[int, str] = {
            bid: " ".join(
                _clean_chunk_text(chunks[i]["text"])
                for i in idxs
                if _clean_chunk_text(chunks[i]["text"])
            )
            for bid, idxs in self.bhajan_chunk_indices.items()
        }

        self.current_bhajan: int = self.bhajan_ids[0]
        self.local_pos: int = 0   # position within current bhajan's chunk list
        self.vote_buffer: deque[int] = deque(maxlen=4)  # 4 = max votes_needed
        self.history: list[str] = []
        dropped = [b for b in (setlist or []) if b not in self.bhajan_ids]
        if dropped:
            print(
                f"[BhajanTracker] WARNING: {len(dropped)} setlist bhajan(s) have no "
                f"chunks and cannot be tracked — IDs {dropped}. "
                "These bhajans have no OCR'd text in the index; rebuild the index or "
                "remove them from the setlist."
            )
        self.setlist: list[int] = [b for b in (setlist or []) if b in self.bhajan_ids]
        self.setlist_ordered: bool = setlist_ordered
        self.setlist_pos: int = 0
        self._low_conf_streak: int = 0  # consecutive rounds position didn't advance
        if self.setlist:
            self.current_bhajan = self.setlist[0]
            order = "ordered" if setlist_ordered else "unordered"
            print(f"[BhajanTracker] Setlist mode ({order}): {len(self.setlist)} bhajans → {self.setlist}")

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _tfidf_score(self, query: str, text: str) -> float:
        """Fuzzy score boosted by discriminative (high-IDF) word overlap."""
        base = fuzz.partial_ratio(query.lower(), text.lower()) / 100.0
        if not self.tfidf_weights:
            return base
        q_words = set(re.findall(r'[\wऀ-ॿ]+', query.lower()))
        t_words = set(re.findall(r'[\wऀ-ॿ]+', text.lower()))
        matching = q_words & t_words
        if not matching:
            return base
        avg_idf = sum(self.tfidf_weights.get(w, 0.0) for w in matching) / len(q_words)
        max_idf = math.log(max(len(self.bhajan_ids), 2))
        bonus = min(0.20, 0.20 * avg_idf / max_idf) if max_idf > 0 else 0.0
        return min(1.0, base + bonus)

    def _window_search_in_bhajan(self, query: str, bhajan_id: int, full: bool = False) -> tuple[int, float]:
        """Return (local_pos, score) within the window around the current position.

        Garbage chunks (clean to empty) are skipped — they produce a false
        ~0.33 fuzzy floor against any Hindi input and prevent streak counting.
        full=True searches the entire bhajan (used for mukhda repeat re-anchoring).
        """
        indices = self.bhajan_chunk_indices.get(bhajan_id, [])
        if not indices:
            return 0, 0.0
        if full:
            lo, hi = 0, len(indices)
        else:
            lo = max(0, self.local_pos - self.WINDOW_BACKWARD)
            hi = min(len(indices), self.local_pos + self.WINDOW_FORWARD)
        best_local, best_score = self.local_pos, 0.0
        for li in range(lo, hi):
            chunk_text = self.chunks[indices[li]]["text"]
            if not _clean_chunk_text(chunk_text):   # skip garbage chunks
                continue
            score = self._tfidf_score(query, chunk_text)
            if score > best_score:
                best_score = score
                best_local = li
        return best_local, best_score

    def _find_best_bhajan(self, query: str, force_global: bool = False) -> tuple[int, float]:
        """Return (bhajan_id, combined_score) for the best matching bhajan.

        force_global=True bypasses setlist filtering (used by the B free-search
        escape when the tracker is stuck and needs to search all bhajans).
        """
        search_ids = (
            self.bhajan_ids if force_global
            else (self.setlist if self.setlist else self.bhajan_ids)
        )

        sem_scores: dict[int, float] = {bid: 0.0 for bid in search_ids}
        if self.faiss_index is not None and self.embed_model is not None:
            try:
                import faiss as _faiss
                q_emb = self.embed_model.encode([query], convert_to_numpy=True)
                _faiss.normalize_L2(q_emb)
                k = min(30, self.faiss_index.ntotal)
                D, I = self.faiss_index.search(q_emb, k)
                for dist, idx in zip(D[0], I[0]):
                    if idx < 0 or idx >= len(self.chunks):
                        continue
                    bid = self.chunks[idx].get("bhajan_id", 0)
                    if bid not in sem_scores:
                        continue  # skip bhajans not in the active search set
                    cosine = (float(dist) + 1.0) / 2.0
                    sem_scores[bid] = max(sem_scores[bid], cosine)
            except Exception as e:
                print("FAISS error in BhajanTracker:", e)

        fuzzy_scores: dict[int, float] = {
            bid: self._tfidf_score(query, self._bhajan_texts[bid])
            for bid in search_ids
        }

        combined = {
            bid: 0.6 * sem_scores[bid] + 0.4 * fuzzy_scores[bid]
            for bid in search_ids
        }

        # Lookahead bonus: give the next bhajan a 15% head-start.
        # Only in ordered setlist mode — unordered mode has no "next" concept.
        if (not force_global and self.setlist_ordered
                and self.setlist and self.setlist_pos + 1 < len(self.setlist)):
            next_bid = self.setlist[self.setlist_pos + 1]
            if next_bid in combined:
                combined[next_bid] = min(1.0, combined[next_bid] + 0.15)

        best_bid = max(combined, key=combined.get)
        return best_bid, combined[best_bid]

    def lock_to_page(self, page: int) -> None:
        """Lock tracker to the chunk nearest to the given global page index."""
        best_bid = self.current_bhajan
        best_li = self.local_pos
        best_dist = float('inf')
        # In setlist mode restrict search to setlist bhajans so a manual scroll
        # into an adjacent bhajan's pages can't silently escape the setlist.
        search_ids = (
            {bid: self.bhajan_chunk_indices[bid] for bid in self.setlist
             if bid in self.bhajan_chunk_indices}
            if self.setlist else self.bhajan_chunk_indices
        )
        for bid, indices in search_ids.items():
            for li, ci in enumerate(indices):
                dist = abs(self.chunks[ci]["page"] - page)
                if dist < best_dist:
                    best_dist = dist
                    best_bid = bid
                    best_li = li
        self.current_bhajan = best_bid
        self.local_pos = best_li
        self.vote_buffer.clear()
        if self.setlist and best_bid in self.setlist:
            self.setlist_pos = self.setlist.index(best_bid)
        print(f"[BhajanTracker] Locked to bhajan {best_bid}, chunk {best_li} (page {page})")

    def _chunk_result(self, bhajan_id: int, local_pos: int) -> tuple[int, str]:
        indices = self.bhajan_chunk_indices[bhajan_id]
        safe = max(0, min(local_pos, len(indices) - 1))
        c = self.chunks[indices[safe]]
        return c["page"], c["text"]

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def process(self, context: str) -> tuple[int, str]:
        """Update state from the latest (possibly multi-chunk) context string.

        Returns (page_index, display_text) for the viewer to scroll to.
        """
        self.history.append(context)
        if len(self.history) > 10:
            self.history.pop(0)

        prev_local_pos = self.local_pos

        # 1. Fast window search within the current bhajan (for position advance)
        local_pos, win_score = self._window_search_in_bhajan(context, self.current_bhajan)

        # 2. ALWAYS run global/setlist search so votes accumulate even during
        #    high-confidence window hits.  If step 2 is skipped when win_score is
        #    high, any shared word like "हरि" in the current bhajan silences all
        #    voting and the tracker can never switch — even when a different bhajan
        #    would score much higher overall.
        best_bid, glob_score = self._find_best_bhajan(context)
        self.vote_buffer.append(best_bid)

        # Position-aware vote threshold (A):
        #   < 20% through bhajan → need 4 votes (guard early false-positives)
        #   20–70%               → need 3 votes (default)
        #   > 70% through bhajan → need 2 votes (responsive near the end)
        total_chunks = max(1, len(self.bhajan_chunk_indices.get(self.current_bhajan, [1])))
        ratio = self.local_pos / (total_chunks - 1) if total_chunks > 1 else 0.0
        if ratio > 0.70:
            votes_needed = 2
        elif ratio < 0.20:
            votes_needed = 4
        else:
            votes_needed = self.SWITCH_VOTES  # 3

        # 3. Switch bhajan only when the last `votes_needed` votes are unanimous
        recent = list(self.vote_buffer)[-votes_needed:]
        if (
            len(recent) == votes_needed
            and len(set(recent)) == 1
            and recent[0] != self.current_bhajan
        ):
            candidate = recent[0]
            can_switch = True
            if self.setlist:
                if self.setlist_ordered:
                    next_pos = self.setlist_pos + 1
                    if next_pos < len(self.setlist) and candidate == self.setlist[next_pos]:
                        self.setlist_pos = next_pos
                    else:
                        can_switch = False
                        self.vote_buffer.clear()
                else:
                    if candidate not in self.setlist:
                        can_switch = False
                        self.vote_buffer.clear()
                    else:
                        self.setlist_pos = self.setlist.index(candidate)

            if can_switch:
                self.current_bhajan = candidate
                self.local_pos = 0
                self.vote_buffer.clear()
                self._low_conf_streak = 0
                print(f"[BhajanTracker] Switched to bhajan {self.current_bhajan}")
                local_pos, _ = self._window_search_in_bhajan(context, self.current_bhajan)
                self.local_pos = local_pos
                return self._chunk_result(self.current_bhajan, self.local_pos)

        # 4. Window is confident → stay and advance regardless of global result.
        #    A 91% window hit means we're in the right place; global disagreement
        #    here would only delay position advance and falsely increment the streak.
        if win_score >= self.HIGH_CONF:
            self._low_conf_streak = 0
            self.local_pos = local_pos
            return self._chunk_result(self.current_bhajan, self.local_pos)

        # 5. Moderate confidence for current bhajan → advance position
        if best_bid == self.current_bhajan and glob_score >= self.LOW_CONF:
            self.local_pos = local_pos

        # A: streak = position not advancing
        if self.local_pos == prev_local_pos:
            self._low_conf_streak += 1
        else:
            self._low_conf_streak = 0

        # D: ordered-setlist auto-advance — stuck at end of current bhajan
        if (
            self.setlist and self.setlist_ordered
            and ratio >= 0.90
            and self._low_conf_streak >= self.STUCK_ROUNDS
            and self.setlist_pos + 1 < len(self.setlist)
        ):
            next_bid = self.setlist[self.setlist_pos + 1]
            self.setlist_pos += 1
            self.current_bhajan = next_bid
            self.local_pos = 0
            self.vote_buffer.clear()
            self._low_conf_streak = 0
            print(f"[BhajanTracker] Auto-advanced → bhajan {next_bid} (end-of-bhajan, {self.STUCK_ROUNDS} stall rounds)")
            return self._chunk_result(self.current_bhajan, 0)

        # B-fast: if another setlist bhajan scores significantly higher than the
        #    current window score, switch immediately without waiting for the streak.
        #    This handles cases like glob=0.67 vs win=0.42 where the answer is clear
        #    but STUCK_ROUNDS delay would add unnecessary seconds of lag.
        if (
            best_bid != self.current_bhajan
            and win_score < self.HIGH_CONF
            and glob_score >= self.HIGH_CONF
            and glob_score > win_score + 0.15
        ):
            can_switch = True
            if self.setlist:
                if self.setlist_ordered:
                    next_pos = self.setlist_pos + 1
                    if not (next_pos < len(self.setlist) and best_bid == self.setlist[next_pos]):
                        can_switch = False
                else:
                    if best_bid not in self.setlist:
                        can_switch = False
                    else:
                        self.setlist_pos = self.setlist.index(best_bid)
            if can_switch:
                print(f"[BhajanTracker] Fast-switch → bhajan {best_bid} "
                      f"(glob={glob_score:.2f} > win={win_score:.2f}+0.15)")
                self.current_bhajan = best_bid
                self.local_pos = 0
                self.vote_buffer.clear()
                self._low_conf_streak = 0
                return self._chunk_result(self.current_bhajan, 0)

        # B: stuck escape — fires after STUCK_ROUNDS of position not advancing,
        #    but only when the window is ALSO unconfident.  If win_score is high we're
        #    already in the right place — don't escape just because position stalled.
        if self._low_conf_streak >= self.STUCK_ROUNDS and win_score < self.HIGH_CONF:
            use_global = not (self.setlist and not self.setlist_ordered)
            free_bid, free_score = self._find_best_bhajan(context, force_global=use_global)
            min_score = self.LOW_CONF if not use_global else self.FREE_SEARCH_CONF
            if free_bid != self.current_bhajan and free_score >= min_score:
                print(f"[BhajanTracker] Escape → bhajan {free_bid} (score={free_score:.2f}, global={use_global})")
                self.current_bhajan = free_bid
                self.local_pos = 0
                self.vote_buffer.clear()
                self._low_conf_streak = 0
                if self.setlist and free_bid in self.setlist:
                    self.setlist_pos = self.setlist.index(free_bid)
                return self._chunk_result(self.current_bhajan, 0)
            elif free_bid == self.current_bhajan:
                # Singer returned to an earlier part (e.g. mukhda repeat) — re-anchor
                # position anywhere in the bhajan, not just the forward window.
                new_pos, new_score = self._window_search_in_bhajan(context, self.current_bhajan, full=True)
                if new_pos != self.local_pos:
                    print(f"[BhajanTracker] Re-anchor pos {self.local_pos}→{new_pos} (mukhda repeat or skip)")
                    self.local_pos = new_pos
                    self._low_conf_streak = 0

        print(
            f"[BhajanTracker] bhajan={self.current_bhajan} "
            f"pos={self.local_pos}/{total_chunks}({ratio:.0%}) "
            f"win={win_score:.2f} glob={glob_score:.2f} "
            f"votes={list(self.vote_buffer)} need={votes_needed} streak={self._low_conf_streak}"
        )
        return self._chunk_result(self.current_bhajan, self.local_pos)


# ---------------------------------------------------------------------------
# Substring extraction helper
# ---------------------------------------------------------------------------

def find_best_match_in_page(page_text: str, query: str) -> str | None:
    if not page_text or not query:
        return None
    q = query.strip()
    idx = page_text.lower().find(q.lower())
    if idx != -1:
        return page_text[idx: idx + len(q)].strip()

    candidates = re.split(r"(?<=[\.!?\n])\s+", page_text)
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

    words = [w for w in re.findall(r"\w+", q)]
    for n in range(len(words), 0, -1):
        for i in range(0, len(words) - n + 1):
            ngram = " ".join(words[i:i + n])
            if ngram and ngram.lower() in choice.lower():
                start = choice.lower().find(ngram.lower())
                if start != -1:
                    return choice[start:start + len(ngram)].strip()

    trimmed = choice.strip()
    if len(trimmed) > 120:
        trimmed = trimmed[:117].rsplit(' ', 1)[0] + '...'
    return trimmed


# ---------------------------------------------------------------------------
# Telegram alerts (stdlib only — no extra dependencies)
# ---------------------------------------------------------------------------

def _send_telegram(token: str, chat_id: str, text: str) -> None:
    """Fire-and-forget Telegram message. Silently fails if offline."""
    if not token or not chat_id:
        return
    try:
        import urllib.request, urllib.parse
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        urllib.request.urlopen(url, data=data, timeout=5)
    except Exception as e:
        print(f"[Telegram] Alert failed: {e}")


def _audio_watchdog(
    audio_loop_fn,
    loop_args: tuple,
    loop_kwargs: dict,
    telegram_token: str = "",
    telegram_chat_id: str = "",
    max_restarts: int = 5,
) -> None:
    """Supervisor that restarts audio_loop on crash and sends Telegram alerts."""
    for attempt in range(max_restarts + 1):
        t = threading.Thread(
            target=audio_loop_fn, args=loop_args, kwargs=loop_kwargs, daemon=True
        )
        t.start()
        t.join()  # blocks until the loop exits (normally or by exception)
        if attempt < max_restarts:
            msg = (
                f"⚠️ Bhajan Live Reader — audio loop crashed, "
                f"restarting ({attempt + 1}/{max_restarts})…"
            )
            print(msg)
            _send_telegram(telegram_token, telegram_chat_id, msg)
            time.sleep(3)
        else:
            msg = "🚨 Bhajan Live Reader — audio loop failed permanently. Please restart the app."
            print(msg)
            _send_telegram(telegram_token, telegram_chat_id, msg)


# ---------------------------------------------------------------------------
# Audio loop
# ---------------------------------------------------------------------------

def _sanitize(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    s = re.sub(r"(?i)\b(?:thankyou|thank you(?: very much)?|thanks(?: a lot)?)\b[\s\.,!]*$", "", s).strip()
    words = s.split()
    # Discard single-word repetition hallucinations (e.g. "रे रे रे रे रे...")
    if len(words) >= 6:
        top_word = max(set(words), key=words.count)
        if words.count(top_word) / len(words) > 0.5:
            return ""
    # Discard phrase-level repetition hallucinations (e.g. "A B C. A B C. A B C.")
    for n in range(2, 7):
        if len(words) < n * 3:
            continue
        phrase = tuple(words[:n])
        chunks = [tuple(words[i:i + n]) for i in range(0, len(words) - n + 1, n)]
        if len(chunks) >= 3 and len(set(chunks)) == 1:
            return ""
    return s


def audio_loop(
    whisper_model,
    viewer,
    pages_text,
    samplerate=16000,
    chunk_s=3,
    embed_model=None,
    faiss_index=None,
    index_meta=None,
    vocab_prompt=None,
    language="hi",
    iw_model=None,
    iw_processor=None,
    position_tracker=None,
):
    use_indicwhisper = iw_model is not None and iw_processor is not None
    import sounddevice as sd
    backend = "IndicWhisper" if use_indicwhisper else "Whisper"
    print(f"Starting audio loop [{backend}] (speak now)...")

    # Overlapping window: keep the tail of the previous chunk so lyrics
    # that fall on a boundary are always captured in at least one window.
    overlap_samples = int(chunk_s * samplerate // 2)
    prev_tail = np.zeros(overlap_samples, dtype='float32')

    # Rolling context: buffer holds up to 7 utterances; normally the last 3
    # are used, expanding to 7 when the tracker is stuck (C).
    context_window: deque[str] = deque(maxlen=7)

    while True:
        try:
            print("Recording...")
            recording = sd.rec(int(chunk_s * samplerate), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()
            new_audio = recording.flatten()
            audio = np.concatenate([prev_tail, new_audio])
            prev_tail = new_audio[-overlap_samples:]

            if not vad_has_speech(audio, samplerate):
                continue

            if use_indicwhisper:
                txt = transcribe_indicwhisper(audio, iw_model, iw_processor, language=language, samplerate=samplerate)
            else:
                transcribe_kwargs: dict = {"language": language, "fp16": False}
                if vocab_prompt:
                    transcribe_kwargs["initial_prompt"] = vocab_prompt
                result = whisper_model.transcribe(audio, **transcribe_kwargs)
                txt = result.get('text', '').strip()

            txt = _sanitize(txt)
            if not txt:
                continue
            print(f"[{backend}] Heard: {txt}")

            context_window.append(txt)
            ctx_size = (position_tracker.context_window_size
                        if position_tracker is not None else 3)
            context = " ".join(list(context_window)[-ctx_size:])

            if position_tracker is not None:
                best_page, display_text = position_tracker.process(context)
                display_text = _clean_chunk_text(display_text)
                if display_text:
                    print(f"[Tracker] → page {best_page}: {display_text[:60]}")
                    # Respect manual-scroll override: the operator may have
                    # paged up/down to show the audience the correct lyrics.
                    # Tracker still runs internally; we just don't scroll the
                    # viewer while the override is active.
                    if time.time() >= getattr(viewer, '_manual_until', 0):
                        viewer.after(0, viewer.highlight_and_scroll, best_page, display_text)
            else:
                best_page, score = hybrid_search(
                    txt, pages_text,
                    embed_model=embed_model,
                    faiss_index=faiss_index,
                    index_meta=index_meta,
                    semantic_weight=0.35,
                )
                print(f"Hybrid search → page {best_page} (score={score:.3f})")
                if score > 0.10:
                    display_text = find_best_match_in_page(pages_text[best_page], txt) or txt
                    viewer.after(0, viewer.highlight_and_scroll, best_page, display_text)

        except Exception as e:
            print("Audio loop error:", e)
            time.sleep(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pdf', required=True)
    p.add_argument('--test', action='store_true', help='Load models only, no GUI/mic')
    p.add_argument('--use_index', action='store_true', help='Enable FAISS semantic retrieval')
    p.add_argument('--build_index', action='store_true', help='Build FAISS index from PDF on start')
    p.add_argument('--language', default='hi', help='Language code for ASR (default: hi)')
    p.add_argument('--chunk_s', type=float, default=3.0, help='Audio chunk length in seconds')

    # ASR backend
    asr_group = p.add_mutually_exclusive_group()
    asr_group.add_argument('--use_indicwhisper', action='store_true',
                           help='Use IndicWhisper (HuggingFace) instead of vanilla Whisper')
    asr_group.add_argument('--whisper_model', default='medium',
                           help='Vanilla Whisper model size: tiny/base/small/medium/large-v3 (default: medium)')
    p.add_argument('--indicwhisper_model', default='ai4bharat/indicwhisper',
                   help='IndicWhisper HuggingFace model ID or path to fine-tuned checkpoint')
    p.add_argument('--gemini_key', default='', help='Gemini API key for position recovery')

    args = p.parse_args()
    pdf_path = args.pdf

    print('Loading PDF...')
    pages_text, images, doc = extract_pages(pdf_path)
    print(f'Extracted {len(pages_text)} pages')

    # --- load ASR model ---
    whisper_model = None
    iw_model = None
    iw_processor = None

    if args.use_indicwhisper:
        iw_model, iw_processor = load_indicwhisper(args.indicwhisper_model)
        print('IndicWhisper ready', flush=True)
    else:
        import whisper
        print(f'Loading vanilla Whisper [{args.whisper_model}]...')
        whisper_model = whisper.load_model(args.whisper_model)
        print('Whisper ready')

    if args.test:
        print('Test mode: models loaded OK')
        return

    # --- build / load FAISS index ---
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

    position_tracker = None
    if args.use_index:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            if Path(idx_path).exists() and Path(meta_path).exists():
                faiss_index = faiss.read_index(idx_path)
                index_meta = np.load(meta_path, allow_pickle=True)
                with open(model_ref, 'r', encoding='utf-8') as f:
                    mm = json.load(f)
                import torch as _torch
                embed_model = SentenceTransformer(
                    mm.get('model', 'paraphrase-multilingual-MiniLM-L12-v2'),
                    device="cuda" if _torch.cuda.is_available() else "cpu")
                print('FAISS index loaded — hybrid retrieval enabled')
                from config import config
                gemini_key = args.gemini_key or config.gemini_api_key
                chunks = [m.item() if hasattr(m, 'item') else m for m in index_meta]
                tfidf_weights: dict = {}
                if Path(f"{base}.tfidf.json").exists():
                    with open(f"{base}.tfidf.json", 'r', encoding='utf-8') as f:
                        tfidf_weights = json.load(f)
                position_tracker = BhajanTracker(
                    chunks=chunks,
                    embed_model=embed_model,
                    faiss_index=faiss_index,
                    tfidf_weights=tfidf_weights,
                    gemini_key=gemini_key,
                )
                n_bhajans = len(set(c.get("bhajan_id", 0) for c in chunks))
                print(f'BhajanTracker ready — {len(chunks)} chunks, {n_bhajans} bhajan(s)')
            else:
                print('Index files not found; run with --build_index first')
        except Exception as e:
            print('Failed to load index:', e)

    vocab_prompt = build_vocab_prompt(pages_text) if not args.use_indicwhisper else None
    if vocab_prompt:
        print(f"Vocab prompt ({len(vocab_prompt)} chars): {vocab_prompt[:80]}...")

    viewer = PDFViewer(images, pages_text, doc=doc)
    t = threading.Thread(
        target=audio_loop,
        args=(whisper_model, viewer, pages_text, 16000, args.chunk_s,
              embed_model, faiss_index, index_meta),
        kwargs={"vocab_prompt": vocab_prompt, "language": args.language,
                "iw_model": iw_model, "iw_processor": iw_processor,
                "position_tracker": position_tracker},
        daemon=True,
    )
    t.start()
    viewer.mainloop()


if __name__ == '__main__':
    import sys
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

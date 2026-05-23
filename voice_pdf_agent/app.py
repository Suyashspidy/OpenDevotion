"""Bhajan Live Reader — setup launcher."""
from __future__ import annotations
import json
import os
import sys
import io

# Allow Devanagari / Unicode text to print on Windows without crashing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

_HERE = Path(__file__).parent

LANGUAGES = {
    "हिन्दी": "hi",
    "संस्कृत": "sa",
    "తెలుగు": "te",
    "தமிழ்": "ta",
    "ಕನ್ನಡ": "kn",
    "বাংলা": "bn",
    "मराठी": "mr",
}

ASR_OPTIONS = {
    "IndicWhisper — best for bhajans": "indicwhisper",
    "Whisper Medium — faster startup": "whisper_medium",
    "Whisper Small — lightest": "whisper_small",
}

_BEST_MODEL = next(
    (p for p in [
        _HERE / "outputs" / "indicwhisper_bhajan",
        _HERE / "outputs" / "indicwhisper_bhajan_v2",
    ] if p.exists()),
    _HERE / "outputs" / "indicwhisper_bhajan",
)
_MAX_PDFS = 3
_MAX_SETLIST = 5
_SETLIST_SAVE = _HERE / "setlist_save.json"


def _find_local_pdfs() -> list[str]:
    return sorted(str(p) for p in _HERE.glob("*.pdf"))


class BhajanSearchDialog(tk.Toplevel):
    """Modal search dialog for picking a bhajan by name."""

    def __init__(self, master, bhajan_options: list, slot_idx: int, on_select):
        super().__init__(master)
        self.title(f"Search Bhajan — Slot {slot_idx + 1}")
        self.resizable(True, True)
        self.grab_set()   # modal
        self._options = bhajan_options
        self._on_select = on_select
        self._filtered: list = []
        self._build_ui()
        self._filter("")

    def _build_ui(self):
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Type to search by bhajan name:").grid(
            row=0, column=0, sticky=tk.W, pady=(0, 4))
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._filter(self._search_var.get()))
        entry = ttk.Entry(frame, textvariable=self._search_var, width=55)
        entry.grid(row=0, column=0, sticky=tk.EW, pady=(18, 6))
        entry.focus_set()

        list_frame = ttk.Frame(frame)
        list_frame.grid(row=1, column=0, sticky=tk.NSEW)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._listbox = tk.Listbox(
            list_frame, width=60, height=18,
            yscrollcommand=scrollbar.set,
            font=("Nirmala UI", 11),   # Devanagari-capable on Windows
            selectmode=tk.SINGLE,
            activestyle="dotbox",
        )
        scrollbar.config(command=self._listbox.yview)
        self._listbox.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)

        self._listbox.bind("<Double-Button-1>", lambda _e: self._select())
        self._listbox.bind("<Return>", lambda _e: self._select())

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, sticky=tk.EW, pady=(8, 0))
        ttk.Button(btn_frame, text="Select", command=self._select).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)

    def _filter(self, query: str):
        q = query.lower().strip()
        self._filtered = [
            opt for opt in self._options
            if not q or q in opt[0].lower()
        ]
        self._listbox.delete(0, tk.END)
        for display, _si, _bid in self._filtered:
            self._listbox.insert(tk.END, display)
        if self._filtered:
            self._listbox.selection_set(0)
            self._listbox.see(0)

    def _select(self):
        sel = self._listbox.curselection()
        if not sel:
            return
        self._on_select(self._filtered[sel[0]][0])
        self.destroy()


class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bhajan Live Reader")
        self.resizable(False, False)
        # (display_str, pdf_slot_idx, local_bhajan_id)
        self._bhajan_options: list[tuple[str, int, int]] = []
        self._build_ui()
        self._populate_mics()
        self._auto_detect()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main = ttk.Frame(self, padding=20)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Bhajan Live Reader",
                  font=("Helvetica", 17, "bold")).grid(
            row=0, column=0, columnspan=3, pady=(0, 2))
        ttk.Label(main, text="Live lyric scrolling for performances",
                  font=("Helvetica", 10), foreground="#666").grid(
            row=1, column=0, columnspan=3, pady=(0, 10))

        row = 2

        # ── PDF Library ──────────────────────────────────────────────
        ttk.Separator(main, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=tk.EW, pady=(6, 4))
        row += 1
        ttk.Label(main, text="PDF Library",
                  font=("Helvetica", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 6))
        row += 1

        self.pdf_vars: list[tk.StringVar] = []
        for i in range(_MAX_PDFS):
            label = "PDF 1 (required):" if i == 0 else f"PDF {i + 1} (optional):"
            ttk.Label(main, text=label).grid(
                row=row, column=0, sticky=tk.W, pady=4)
            var = tk.StringVar()
            self.pdf_vars.append(var)
            ttk.Entry(main, textvariable=var, width=44).grid(
                row=row, column=1, padx=8)
            ttk.Button(main, text="Browse…",
                       command=lambda idx=i: self._browse_pdf(idx)).grid(
                row=row, column=2)
            row += 1

        ttk.Button(main, text="Detect Bhajans ▼",
                   command=self._detect_bhajans).grid(
            row=row, column=2, pady=(2, 0))
        row += 1

        # ── Setlist ──────────────────────────────────────────────────
        ttk.Separator(main, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=tk.EW, pady=(10, 4))
        row += 1
        ttk.Label(main, text="Setlist  (optional)",
                  font=("Helvetica", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 2))
        row += 1
        ttk.Label(main,
                  text="Pick up to 5 bhajans. Leave blank to follow freely.",
                  font=("Helvetica", 9), foreground="#666").grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 6))
        row += 1

        self.setlist_vars: list[tk.StringVar] = []
        self.setlist_combos: list[ttk.Combobox] = []
        for i in range(_MAX_SETLIST):
            ttk.Label(main, text=f"{i + 1}.").grid(
                row=row, column=0, sticky=tk.E, pady=3)
            var = tk.StringVar(value="(none)")
            combo = ttk.Combobox(main, textvariable=var, width=44,
                                 state="readonly", values=["(none)"])
            combo.grid(row=row, column=1, padx=8, sticky=tk.W)
            ttk.Button(main, text="🔍 Search",
                       command=lambda idx=i: self._open_bhajan_search(idx)).grid(
                row=row, column=2, sticky=tk.W)
            self.setlist_vars.append(var)
            self.setlist_combos.append(combo)
            row += 1

        self.setlist_ordered_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            main,
            text="Enforce performance order  (1 → 2 → 3 → …  tracker never goes backwards)",
            variable=self.setlist_ordered_var,
        ).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))
        row += 1

        # ── Audio / ASR settings ─────────────────────────────────────
        ttk.Separator(main, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=tk.EW, pady=(10, 4))
        row += 1

        ttk.Label(main, text="Microphone:").grid(
            row=row, column=0, sticky=tk.W, pady=6)
        self.mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(main, textvariable=self.mic_var,
                                      width=44, state="readonly")
        self.mic_combo.grid(row=row, column=1, padx=8)
        ttk.Button(main, text="Refresh",
                   command=self._populate_mics).grid(row=row, column=2)
        row += 1

        ttk.Label(main, text="Language:").grid(
            row=row, column=0, sticky=tk.W, pady=6)
        self.lang_var = tk.StringVar(value="हिन्दी")
        ttk.Combobox(main, textvariable=self.lang_var,
                     values=list(LANGUAGES.keys()),
                     width=24, state="readonly").grid(
            row=row, column=1, padx=8, sticky=tk.W)
        row += 1

        ttk.Label(main, text="ASR Model:").grid(
            row=row, column=0, sticky=tk.W, pady=6)
        self.asr_var = tk.StringVar(value=list(ASR_OPTIONS.keys())[0])
        ttk.Combobox(main, textvariable=self.asr_var,
                     values=list(ASR_OPTIONS.keys()),
                     width=44, state="readonly").grid(
            row=row, column=1, padx=8)
        row += 1

        ttk.Label(main, text="Fine-tuned model\n(optional):").grid(
            row=row, column=0, sticky=tk.W, pady=6)
        self.model_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.model_var, width=44).grid(
            row=row, column=1, padx=8)
        ttk.Button(main, text="Browse…",
                   command=self._browse_model).grid(row=row, column=2)
        row += 1

        ttk.Separator(main, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=tk.EW, pady=(10, 10))
        row += 1

        self.start_btn = ttk.Button(
            main, text="▶  Start Session", command=self._start, width=22)
        self.start_btn.grid(row=row, column=0, columnspan=3)
        row += 1

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(main, textvariable=self.status_var,
                  foreground="#666", font=("Helvetica", 9)).grid(
            row=row, column=0, columnspan=3, pady=(10, 0))

    # ------------------------------------------------------------------
    # Setlist persistence
    # ------------------------------------------------------------------

    def _save_setlist(self):
        data = {
            "pdf_paths":      [v.get().strip() for v in self.pdf_vars],
            "setlist_slots":  [v.get().strip() for v in self.setlist_vars],
            "setlist_ordered": self.setlist_ordered_var.get(),
        }
        try:
            with open(_SETLIST_SAVE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[setlist save] Could not save: {e}")

    def _restore_setlist(self):
        if not _SETLIST_SAVE.exists():
            return
        try:
            with open(_SETLIST_SAVE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[setlist save] Could not load: {e}")
            return

        # Restore PDF paths only if the saved paths still exist on disk
        for i, path in enumerate(data.get("pdf_paths", [])[:_MAX_PDFS]):
            if path and Path(path).exists() and not self.pdf_vars[i].get().strip():
                self.pdf_vars[i].set(path)

        # Re-detect bhajans so combo values are populated before restoring slots
        self._detect_bhajans()

        # Restore setlist slot values — only if they exist in the current options
        valid = {o[0] for o in self._bhajan_options}
        for i, slot_val in enumerate(data.get("setlist_slots", [])[:_MAX_SETLIST]):
            if slot_val and slot_val != "(none)" and slot_val in valid:
                self.setlist_vars[i].set(slot_val)

        if data.get("setlist_ordered") is not None:
            self.setlist_ordered_var.set(bool(data["setlist_ordered"]))

        saved = sum(1 for s in data.get("setlist_slots", [])
                    if s and s != "(none)")
        if saved:
            self.status_var.set(
                f"Restored {saved} bhajan(s) from last session. "
                "Start or edit the setlist.")

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    def _auto_detect(self):
        pdfs = _find_local_pdfs()
        for i, path in enumerate(pdfs[:_MAX_PDFS]):
            self.pdf_vars[i].set(path)
        self.model_var.set(str(_BEST_MODEL))
        # Restore saved setlist (calls _detect_bhajans internally)
        self._restore_setlist()
        # If no save file, still populate dropdowns
        if not _SETLIST_SAVE.exists():
            self._detect_bhajans()

    # ------------------------------------------------------------------
    # Mic enumeration
    # ------------------------------------------------------------------

    def _populate_mics(self):
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            inputs = [f"{i}: {d['name']}" for i, d in enumerate(devs)
                      if d["max_input_channels"] > 0]
            self.mic_combo["values"] = inputs
            if inputs:
                try:
                    default_idx = sd.default.device[0]
                    match = next(
                        (d for d in inputs if d.startswith(f"{default_idx}:")),
                        inputs[0])
                    self.mic_var.set(match)
                except Exception:
                    self.mic_var.set(inputs[0])
            else:
                self.mic_var.set("")
        except Exception as e:
            messagebox.showerror(
                "Microphone Error",
                f"Could not list microphones:\n{e}\n\nEnsure sounddevice is installed.")

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

    def _browse_pdf(self, idx: int):
        path = filedialog.askopenfilename(
            title=f"Select PDF {idx + 1}",
            initialdir=str(_HERE),
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self.pdf_vars[idx].set(path)

    def _browse_model(self):
        path = filedialog.askdirectory(
            title="Select Fine-tuned Model Folder",
            initialdir=str(_HERE / "outputs"),
        )
        if path:
            self.model_var.set(path)

    # ------------------------------------------------------------------
    # Bhajan detection (populates setlist dropdowns)
    # ------------------------------------------------------------------

    def _detect_bhajans(self):
        options: list[tuple[str, int, int]] = []

        for slot_idx, var in enumerate(self.pdf_vars):
            pdf_path = var.get().strip()
            if not pdf_path or not Path(pdf_path).exists():
                continue
            base = Path(pdf_path).stem
            pdf_dir = Path(pdf_path).parent

            # Prefer pre-built bhajans.json (fast, has proper titles)
            bjpath = pdf_dir / f"{base}.bhajans.json"
            if bjpath.exists():
                try:
                    with open(bjpath, "r", encoding="utf-8") as f:
                        bjdata = json.load(f)
                    for bj in bjdata:
                        display = f"{bj['title'][:45]}  [{base}]"
                        options.append((display, slot_idx, bj["bhajan_id"]))
                    continue
                except Exception:
                    pass

            # Fallback: scan index_meta.npy for bhajan_ids
            meta_path = pdf_dir / f"{base}.index_meta.npy"
            if meta_path.exists():
                try:
                    meta = np.load(str(meta_path), allow_pickle=True)
                    chunks = [m.item() if hasattr(m, "item") else m for m in meta]
                    seen: dict[int, str] = {}
                    for c in chunks:
                        bid = c.get("bhajan_id", 0)
                        if bid not in seen:
                            seen[bid] = c["text"][:45]
                    for bid, title in sorted(seen.items()):
                        display = f"{title}  [{base}]"
                        options.append((display, slot_idx, bid))
                except Exception:
                    pass

        self._bhajan_options = options
        display_strs = ["(none)"] + [o[0] for o in options]
        for combo in self.setlist_combos:
            combo["values"] = display_strs
            if combo.get() not in display_strs:
                combo.set("(none)")

        n_pdfs = sum(1 for v in self.pdf_vars if v.get().strip()
                     and Path(v.get().strip()).exists())
        self.status_var.set(
            f"Found {len(options)} bhajans across {n_pdfs} PDF(s). "
            "Select your setlist above.")

    # ------------------------------------------------------------------
    # Bhajan search dialog
    # ------------------------------------------------------------------

    def _open_bhajan_search(self, slot_idx: int):
        if not self._bhajan_options:
            messagebox.showinfo(
                "No Bhajans",
                "No bhajans detected yet.\nPlease browse a PDF first, then click Detect Bhajans.")
            return

        def _on_select(display_str: str):
            self.setlist_vars[slot_idx].set(display_str)

        BhajanSearchDialog(self, self._bhajan_options, slot_idx, _on_select)

    # ------------------------------------------------------------------
    # Start session
    # ------------------------------------------------------------------

    def _start(self):
        pdf1 = self.pdf_vars[0].get().strip()
        if not pdf1:
            messagebox.showwarning("Missing PDF", "Please select at least PDF 1.")
            return
        if not Path(pdf1).exists():
            messagebox.showerror("File Not Found", f"PDF 1 not found:\n{pdf1}")
            return
        if not self.mic_var.get():
            messagebox.showwarning(
                "No Microphone",
                "No microphone detected.\nPlease connect a mic and click Refresh.")
            return

        asr_mode = ASR_OPTIONS[self.asr_var.get()]
        lang_code = LANGUAGES.get(self.lang_var.get(), "hi")
        mic_index = int(self.mic_var.get().split(":")[0])
        model_path = self.model_var.get().strip() or None

        # Read all tkinter vars from the main thread before handing off
        pdf_paths = [v.get().strip() for v in self.pdf_vars]
        setlist_selections = [v.get().strip() for v in self.setlist_vars]
        setlist_ordered = self.setlist_ordered_var.get()

        self._save_setlist()
        self.start_btn.config(state=tk.DISABLED)
        self.status_var.set("Loading — please wait…")

        threading.Thread(
            target=self._load_and_launch,
            args=(pdf_paths, setlist_selections, asr_mode,
                  lang_code, mic_index, model_path, setlist_ordered),
            daemon=True,
        ).start()

    def _load_and_launch(self, pdf_paths, setlist_selections,
                         asr_mode, lang_code, mic_index, model_path,
                         setlist_ordered: bool = False):
        try:
            import sounddevice as sd
            sd.default.device = (mic_index, None)

            sys.path.insert(0, str(_HERE))
            from live_search import (
                audio_loop, build_vocab_prompt, extract_pages,
                load_indicwhisper, PDFViewer, BhajanTracker,
            )

            # ── Load all PDFs ──────────────────────────────────────
            self.after(0, self.status_var.set, "Loading PDFs…")
            valid_paths = [p for p in pdf_paths
                           if p and Path(p).exists()]

            all_pages_text: list[str] = []
            all_images: list = []
            all_docs: list = []
            page_offsets: list[int] = []
            current_page_offset = 0

            for pdf_path in valid_paths:
                pages_text, images, doc = extract_pages(pdf_path)
                page_offsets.append(current_page_offset)
                all_pages_text.extend(pages_text)
                all_images.extend(images)
                all_docs.append(doc)
                current_page_offset += len(pages_text)

            # ── Pre-load VAD ───────────────────────────────────────
            self.after(0, self.status_var.set, "Loading VAD (silero)…")
            try:
                from live_search import _vad_load
                _vad_load()
            except Exception as e:
                print(f"VAD pre-load skipped: {e}")

            # ── Load ASR model ─────────────────────────────────────
            whisper_model = iw_model = iw_processor = None
            if asr_mode == "indicwhisper":
                model_id = model_path or "ai4bharat/indicwhisper"
                self.after(0, self.status_var.set,
                           f"Loading IndicWhisper from {Path(model_id).name}…")
                iw_model, iw_processor = load_indicwhisper(model_id)
            else:
                import whisper
                size = "medium" if asr_mode == "whisper_medium" else "small"
                self.after(0, self.status_var.set, f"Loading Whisper {size}…")
                whisper_model = whisper.load_model(size)

            # ── Load and merge FAISS indexes ───────────────────────
            self.after(0, self.status_var.set, "Loading search indexes…")
            merged_chunks: list[dict] = []
            merged_vectors: list[np.ndarray] = []
            merged_embed_model = None
            merged_faiss_index = None
            current_bhajan_offset = 0
            # maps (slot_idx, local_bhajan_id) → global_bhajan_id
            local_to_global: dict[tuple[int, int], int] = {}

            for slot_idx, pdf_path in enumerate(valid_paths):
                base = Path(pdf_path).stem
                pdf_dir = Path(pdf_path).parent
                idx_path  = pdf_dir / f"{base}.index.faiss"
                meta_path = pdf_dir / f"{base}.index_meta.npy"
                model_ref = pdf_dir / f"{base}.embed_model.json"

                if not (idx_path.exists() and meta_path.exists()):
                    print(f"No index for {base}, skipping FAISS for this PDF.")
                    continue

                try:
                    import faiss
                    from sentence_transformers import SentenceTransformer

                    pdf_index = faiss.read_index(str(idx_path))
                    pdf_meta  = np.load(str(meta_path), allow_pickle=True)
                    pdf_chunks = [m.item() if hasattr(m, "item") else m
                                  for m in pdf_meta]

                    if merged_embed_model is None and model_ref.exists():
                        with open(model_ref, "r", encoding="utf-8") as f:
                            mm = json.load(f)
                        import torch as _torch
                        _st_device = "cuda" if _torch.cuda.is_available() else "cpu"
                        merged_embed_model = SentenceTransformer(
                            mm.get("model",
                                   "paraphrase-multilingual-MiniLM-L12-v2"),
                            device=_st_device)

                    local_bids = sorted(
                        set(c.get("bhajan_id", 0) for c in pdf_chunks))
                    for local_bid in local_bids:
                        local_to_global[(slot_idx, local_bid)] = (
                            current_bhajan_offset + local_bid)
                    max_local = max(local_bids) if local_bids else 0
                    current_bhajan_offset += max_local + 1

                    page_off = page_offsets[slot_idx]
                    for chunk in pdf_chunks:
                        adj = dict(chunk)
                        adj["page"] = chunk["page"] + page_off
                        adj["bhajan_id"] = local_to_global[
                            (slot_idx, chunk.get("bhajan_id", 0))]
                        merged_chunks.append(adj)

                    # Extract raw vectors for merging.
                    # vector_to_array is O(1) for IndexFlat*; reconstruct is
                    # a fallback for other index types.
                    dim = pdf_index.d
                    ntotal = pdf_index.ntotal
                    try:
                        raw = faiss.vector_to_array(pdf_index.xb)
                        vecs = np.array(raw, dtype="float32").reshape(ntotal, dim)
                    except (AttributeError, Exception):
                        vecs = np.vstack(
                            [pdf_index.reconstruct(i).reshape(1, -1)
                             for i in range(ntotal)])
                    merged_vectors.append(vecs)

                except Exception as e:
                    import traceback
                    print(f"Index load failed for {Path(pdf_path).name}: {e}")
                    traceback.print_exc()

            if merged_vectors and merged_embed_model is not None:
                import faiss as _faiss
                all_vecs = np.concatenate(merged_vectors, axis=0)
                dim = all_vecs.shape[1]
                merged_faiss_index = _faiss.IndexFlatIP(dim)
                merged_faiss_index.add(all_vecs)

            # TF-IDF weights across merged chunks
            merged_tfidf: dict = {}
            if merged_chunks:
                from build_index import build_tfidf_weights
                merged_tfidf = build_tfidf_weights(merged_chunks)

            # ── Build setlist ──────────────────────────────────────
            setlist: list[int] = []
            for sel in setlist_selections:
                if not sel or sel == "(none)":
                    continue
                for display, slot_idx, local_bid in self._bhajan_options:
                    if display == sel:
                        global_bid = local_to_global.get((slot_idx, local_bid))
                        if global_bid is not None and global_bid not in setlist:
                            setlist.append(global_bid)
                        break

            # ── Build BhajanTracker ────────────────────────────────
            position_tracker = None
            if merged_chunks:
                from config import config as _cfg
                position_tracker = BhajanTracker(
                    chunks=merged_chunks,
                    embed_model=merged_embed_model,
                    faiss_index=merged_faiss_index,
                    tfidf_weights=merged_tfidf,
                    gemini_key=_cfg.gemini_api_key,
                    setlist=setlist or None,
                    setlist_ordered=setlist_ordered,
                )
                n_bhajans = len(
                    set(c.get("bhajan_id", 0) for c in merged_chunks))
                if setlist:
                    order = "ordered" if setlist_ordered else "unordered"
                    mode = f"{order} setlist [{len(setlist)} bhajans]"
                else:
                    mode = "free mode"
                print(f"BhajanTracker ready — {len(merged_chunks)} chunks, "
                      f"{n_bhajans} bhajan(s), {mode}")

            vocab_prompt = (
                build_vocab_prompt(all_pages_text)
                if asr_mode != "indicwhisper" else None)

            primary_doc = all_docs[0] if all_docs else None
            self.after(0, self._open_viewer,
                       all_pages_text, all_images, primary_doc,
                       whisper_model, iw_model, iw_processor,
                       merged_embed_model, merged_faiss_index, None,
                       vocab_prompt, lang_code, position_tracker)

        except Exception as e:
            import traceback
            detail = traceback.format_exc()
            self.after(0, messagebox.showerror, "Launch Error",
                       f"{type(e).__name__}: {e}\n\n{detail}")
            self.after(0, self._reset_ui)

    def _reset_ui(self):
        self.start_btn.config(state=tk.NORMAL)
        self.status_var.set("Ready.")

    def _open_viewer(self, pages_text, images, doc,
                     whisper_model, iw_model, iw_processor,
                     embed_model, faiss_index, index_meta,
                     vocab_prompt, lang_code, position_tracker=None):
        from live_search import PDFViewer, audio_loop, _audio_watchdog, _send_telegram
        from config import config as _cfg

        tg_token = _cfg.telegram_bot_token
        tg_chat  = _cfg.telegram_chat_id

        self.withdraw()

        viewer = PDFViewer(images, pages_text, doc=doc, master=self,
                           position_tracker=position_tracker)
        viewer.title("Bhajan Live Reader — Live Session")

        def _on_close():
            viewer.destroy()
            self.deiconify()
            self._reset_ui()

        viewer.protocol("WM_DELETE_WINDOW", _on_close)

        loop_args = (whisper_model, viewer, pages_text, 16000, 3.0,
                     embed_model, faiss_index, index_meta)
        loop_kwargs = {"vocab_prompt": vocab_prompt, "language": lang_code,
                       "iw_model": iw_model, "iw_processor": iw_processor,
                       "position_tracker": position_tracker}

        threading.Thread(
            target=_audio_watchdog,
            args=(audio_loop, loop_args, loop_kwargs),
            kwargs={"telegram_token": tg_token, "telegram_chat_id": tg_chat},
            daemon=True,
        ).start()

        # Startup ping so you know the session is live
        _send_telegram(tg_token, tg_chat, "✅ Bhajan Live Reader — session started.")

        # Auto-scroll and lock to first setlist bhajan so user doesn't have to
        if position_tracker is not None and position_tracker.setlist:
            first_bid = position_tracker.setlist[0]
            indices = position_tracker.bhajan_chunk_indices.get(first_bid, [])
            if indices:
                start_page = position_tracker.chunks[indices[0]]["page"]

                def _auto_lock():
                    viewer.scroll_to_page(start_page)
                    position_tracker.lock_to_page(start_page)
                    if hasattr(viewer, "_lock_btn"):
                        viewer._lock_btn.config(
                            text=f"📍  Locked — page {start_page + 1}",
                            bg='#1a5c1a')

                viewer.after(800, _auto_lock)

        self.status_var.set("Session running.")


def main():
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()

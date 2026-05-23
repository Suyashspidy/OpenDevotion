"""Stack-level diagnostic: simulates exactly what live_search.py does before the GUI."""
import sys, traceback

def step(msg):
    print(msg, flush=True)

try:
    step("1. importing fitz / PIL / tkinter...")
    import fitz
    from PIL import Image, ImageTk
    import tkinter as tk

    step("2. opening PDF and rendering 202 pages...")
    doc = fitz.open("Sankirtan Madhuri.pdf")
    images = []
    for p in doc:
        pix = p.get_pixmap(alpha=False)
        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    step(f"   rendered {len(images)} pages")

    step("3. loading WhisperProcessor + WhisperForConditionalGeneration...")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained("outputs/indicwhisper_bhajan")
    model = WhisperForConditionalGeneration.from_pretrained("outputs/indicwhisper_bhajan")
    model.eval()
    step("   model loaded")

    step("4. importing faiss...")
    import faiss
    step("   faiss ok")

    step("5. importing SentenceTransformer...")
    from sentence_transformers import SentenceTransformer
    step("   import ok")

    step("6. loading sentence transformer model...")
    import json
    from pathlib import Path
    with open("Sankirtan Madhuri.embed_model.json") as f:
        mm = json.load(f)
    embed_model = SentenceTransformer(mm.get("model", "paraphrase-multilingual-MiniLM-L12-v2"))
    step("   model loaded")

    step("7. reading FAISS index...")
    faiss_index = faiss.read_index("Sankirtan Madhuri.index.faiss")
    import numpy as np
    index_meta = np.load("Sankirtan Madhuri.index_meta.npy", allow_pickle=True)
    step(f"   index has {faiss_index.ntotal} vectors")

    step("8. creating tkinter PDFViewer window...")
    root = tk.Tk()
    root.title("Diagnostic")
    root.after(1000, root.destroy)   # auto-close after 1 s
    root.mainloop()
    step("   window opened and closed ok")

    step("ALL STEPS PASSED")

except Exception:
    step("CRASH at step above:")
    traceback.print_exc()

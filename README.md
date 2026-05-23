# OpenDevotion

An AI-powered real-time lyric assistant for live bhajan and keertan performances. Listens to the microphone, identifies the bhajan being sung using a semantic FAISS index built from your PDF songbook, and scrolls the PDF to the correct page automatically.

The IndicWhisper model is fine-tuned specifically on Hindi and Sanskrit bhajan audio using AMD GPU cloud credits, giving it significantly higher accuracy on devotional music than a general-purpose speech model.

---

## Features

- **Real-time ASR** — IndicWhisper fine-tuned on bhajan audio via AMD GPU cloud training, or standard Whisper Medium/Small
- **Semantic search** — FAISS index over your PDF songbooks; handles OCR-scanned Hindi/Sanskrit books
- **Auto page tracking** — live PDF viewer scrolls to the matching page as the singing progresses
- **Setlist mode** — pre-load up to 5 bhajans; optional ordered progression (1 → 2 → 3 …)
- **7 Indian languages** — हिन्दी · संस्कृत · తెలుగు · தமிழ் · ಕನ್ನಡ · বাংলা · मराठी
- **Telegram alerts** — optional push notification on each page change (for stage/backstage co-ordination)
- **Windows desktop GUI** — single Tkinter launcher, no browser required

---

## Requirements

| Dependency | Notes |
|---|---|
| Python 3.10+ | |
| CUDA GPU (recommended) | RTX class; CPU mode works but is slow |
| [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) | Required for indexing scanned PDFs |
| ffmpeg | Required for audio processing; add to PATH |

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd OpenDevotion
python -m venv .venv
.venv\Scripts\activate
pip install -r voice_pdf_agent/requirements.txt
```

### 2. Environment variables (optional)

Create `voice_pdf_agent/.env` for Telegram integration:

```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. Build the index

Place your bhajan PDF(s) inside `voice_pdf_agent/` and run once per PDF:

```bash
cd voice_pdf_agent
python build_index.py "Your Songbook.pdf" --ocr
python remap_bhajans.py "Your Songbook.pdf"
```

> The `--ocr` flag is required for scanned PDFs (most printed Indian devotional books).  
> `remap_bhajans.py` must be run after every `build_index.py` call.

### 4. Launch

```bash
python voice_pdf_agent/app.py
```

---

## How to use

1. Load one or more PDFs with the **Browse…** buttons.
2. Click **Detect Bhajans ▼** to scan the PDFs and populate the bhajan list.
3. *(Optional)* Build a setlist of up to 5 bhajans using the search buttons.
4. Select the correct **Microphone** and **Language**.
5. Click **▶ Start Session** — the lyric window opens and follows the singing in real time.

---

## Project layout

```
voice_pdf_agent/
├── app.py                  # GUI launcher (run this)
├── live_search.py          # Real-time ASR + FAISS matching engine
├── build_index.py          # One-time index builder per PDF
├── remap_bhajans.py        # Bhajan name/page remapper (run after build_index)
├── config.py               # Env-based configuration
├── train_indicwhisper.py   # Fine-tuning script (AMD/CUDA cloud training)
├── requirements.txt
└── *.pdf                   # Place your songbooks here
```

---

## ASR model options

| Option | Best for |
|---|---|
| IndicWhisper — best for bhajans | Fine-tuned on Hindi/Sanskrit bhajan audio; highest accuracy |
| Whisper Medium — faster startup | Good general Hindi; loads in ~10 s |
| Whisper Small — lightest | Low-RAM machines; lower accuracy |

A fine-tuned IndicWhisper checkpoint can be pointed to via the **Fine-tuned model** field in the launcher.

---

## License

MIT

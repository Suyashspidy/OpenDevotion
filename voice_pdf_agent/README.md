# OpenDevotion — voice_pdf_agent

Real-time ASR + PDF tracking engine for live devotional performances.

## Quick start

1. Create and activate a Python 3.10+ virtual env

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your API keys (see `config.py`).

3. Build the index for your scripture/songbook PDF:

```bash
python build_index.py "Your Scripture.pdf" --ocr
python remap_bhajans.py "Your Scripture.pdf"
```

4. Launch the app:

```bash
python app.py
```

## Project layout

```
voice_pdf_agent/
├── app.py                  # GUI launcher (run this)
├── live_search.py          # Real-time ASR + FAISS matching engine
├── build_index.py          # One-time index builder per PDF
├── remap_bhajans.py        # Hymn name/page remapper (run after build_index)
├── config.py               # Env-based configuration
├── train_indicwhisper.py   # IndicWhisper fine-tuning script
├── train_wav2vec2.py       # Wav2Vec2 fine-tuning script
├── requirements.txt
└── *.pdf                   # Place your scripture PDFs here
```

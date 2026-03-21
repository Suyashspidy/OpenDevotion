# VoiceSearch PDF Agent (Claude API)

Minimal MVP to speak a phrase and highlight matches inside a PDF.

Quick start

1. Create and activate a Python 3.10+ virtual env

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# install
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set `CLAUDE_API_KEY`.

3. Usage (CLI MVP):

```bash
python app.py --pdf path/to/doc.pdf --audio query.wav
```

Project layout

```
/voice_pdf_agent
├── app.py
├── config.py
├── speech_to_text.py
├── pdf_reader.py
├── matcher.py
├── highlighter.py
├── agent.py
├── requirements.txt
└── README.md
```

Next steps

- Implement `speech_to_text.py`, `pdf_reader.py`, `matcher.py`, `highlighter.py`, and `agent.py` incrementally.
- Add tests and a small demo PDF/audio.

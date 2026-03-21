"""CLI entrypoint for VoiceSearch PDF Agent (MVP).

Usage:
    python app.py --pdf document.pdf --audio query.wav

This is a minimal stub that wires modules together; modules will be implemented step-by-step.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from config import config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VoiceSearch PDF Agent CLI")
    p.add_argument("--pdf", required=True, help="Path to input PDF")
    p.add_argument("--audio", required=True, help="Path to query audio (wav/mp3)")
    p.add_argument("--out", default=None, help="Path for highlighted output PDF")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf)
    audio_path = Path(args.audio)
    out_path = Path(args.out) if args.out else None

    # TODO: Implement the end-to-end flow using modules:
    # 1. transcribe = speech_to_text.transcribe(audio_path)
    # 2. pages = pdf_reader.extract_text(pdf_path)
    # 3. results = agent.search_and_decide(transcribe, pages)
    # 4. highlighter.apply_highlights(pdf_path, results, out_path)

    print("Stub: wired CLI. Implement modules next.")


if __name__ == "__main__":
    main()

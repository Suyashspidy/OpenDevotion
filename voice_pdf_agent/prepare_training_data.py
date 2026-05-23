"""Prepare bhajan training data from a list of YouTube URLs.

For each URL the script:
  1. Downloads audio as 16 kHz mono WAV via yt-dlp
  2. Downloads subtitles (manual first, auto-generated as fallback)
  3. Parses the VTT file into (start, end, text) segments
  4. Deduplicates consecutive identical lines (common in auto-captions)
  5. Cuts the audio at those timestamps using pydub
  6. Writes a CSV ready for train_indicwhisper.py

Requirements (besides requirements.txt):
  - yt-dlp  : pip install yt-dlp
  - ffmpeg  : must be on PATH (https://ffmpeg.org/download.html)

Usage:
  python prepare_training_data.py \
    --urls urls.txt \
    --output_dir data/bhajan_segments \
    --out_csv bhajan_train.csv

urls.txt format  (one URL per line, # for comments):
  https://youtube.com/watch?v=abc123
  https://youtube.com/watch?v=def456   # Hare Krishna bhajan
"""
from __future__ import annotations
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# allow Devanagari characters to print on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# invoke yt-dlp via the current Python interpreter so PATH is not required
YTDLP = [sys.executable, "-m", "yt_dlp"]

def _ffmpeg_location() -> str:
    """Return path to the ffmpeg binary bundled with imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # fall back to system ffmpeg if available


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Prepare bhajan training CSV from YouTube URLs")
    p.add_argument("--urls", required=True, help="Text file with YouTube URLs, one per line")
    p.add_argument("--output_dir", default="data/bhajan_segments", help="Root dir for downloaded files and segments")
    p.add_argument("--out_csv", default="bhajan_train.csv")
    p.add_argument("--lang", default="hi", help="Subtitle language code (default: hi)")
    p.add_argument("--min_duration_ms", type=int, default=500,  help="Drop segments shorter than this")
    p.add_argument("--max_duration_ms", type=int, default=15000, help="Drop segments longer than this")
    p.add_argument("--padding_ms", type=int, default=100, help="Extra audio padding around each segment")
    return p.parse_args()


# ---------------------------------------------------------------------------
# URL file reader
# ---------------------------------------------------------------------------

def read_urls(path: str) -> list[str]:
    urls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                urls.append(line)
    return urls


# ---------------------------------------------------------------------------
# VTT parsing
# ---------------------------------------------------------------------------

def _vtt_time_to_ms(t: str) -> int:
    """Convert VTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to milliseconds."""
    t = t.strip().split()[0]            # drop any positioning metadata
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = parts
    else:
        h, (m, s) = "0", parts
    sec, ms_str = s.split(".")
    ms = int(ms_str[:3].ljust(3, "0"))  # normalise to exactly 3 digits
    return int(h) * 3_600_000 + int(m) * 60_000 + int(sec) * 1_000 + ms


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)             # remove inline VTT tags
    text = re.sub(r"[♪♫♬🎵🎶]", "", text)           # remove music symbols
    text = re.sub(r"\[.*?\]", "", text)              # remove [Music], [Applause] etc.
    text = " ".join(text.split())                    # normalise whitespace
    return text.strip()


def parse_vtt(vtt_path: str) -> list[tuple[int, int, str]]:
    """Return list of (start_ms, end_ms, text) from a VTT subtitle file."""
    with open(vtt_path, encoding="utf-8") as f:
        content = f.read()

    segments: list[tuple[int, int, str]] = []
    for block in re.split(r"\n\n+", content):
        lines = block.strip().splitlines()
        timing_line = None
        text_lines: list[str] = []
        for line in lines:
            if "-->" in line:
                timing_line = line
            elif timing_line and line and not line.strip().isdigit() and not line.startswith("WEBVTT"):
                text_lines.append(line)

        if not timing_line or not text_lines:
            continue

        try:
            left, right = timing_line.split("-->")
            start_ms = _vtt_time_to_ms(left)
            end_ms = _vtt_time_to_ms(right)
        except Exception:
            continue

        text = _clean_text(" ".join(text_lines))
        if text:
            segments.append((start_ms, end_ms, text))

    return segments


def deduplicate_segments(segments: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Merge consecutive identical lines — a common artefact of auto-captions."""
    if not segments:
        return []
    result = [list(segments[0])]
    for start, end, text in segments[1:]:
        if text == result[-1][2]:
            result[-1][1] = end          # extend end time instead of adding duplicate
        else:
            result.append([start, end, text])
    return [tuple(r) for r in result]   # type: ignore[return-value]


# ---------------------------------------------------------------------------
# yt-dlp download
# ---------------------------------------------------------------------------

def _get_video_id(url: str) -> str | None:
    """Extract YouTube video ID directly from the URL (no subprocess needed)."""
    parsed = urlparse(url)
    # standard: youtube.com/watch?v=ID
    params = parse_qs(parsed.query)
    if "v" in params:
        return params["v"][0]
    # short URL: youtu.be/ID
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/").split("/")[0]
    return None


def download_description(url: str) -> str:
    """Return the video description text, or empty string on failure."""
    result = subprocess.run(
        [*YTDLP, "--no-playlist", "--print", "description", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def extract_devanagari_lines(text: str) -> list[str]:
    """Extract non-empty lines that contain Devanagari script (Hindi/Sanskrit lyrics)."""
    lines = []
    for line in text.splitlines():
        line = _clean_text(line)
        if line and re.search(r"[ऀ-ॿ]", line) and len(line) > 4:
            lines.append(line)
    return lines


def align_with_description(
    segments: list[tuple[int, int, str]],
    desc_lines: list[str],
    threshold: int = 55,
) -> list[tuple[int, int, str]]:
    """Replace each subtitle transcript with the best-matching description line.

    Subtitles give accurate timestamps; description gives clean text.
    Only replaces when fuzzy score >= threshold — otherwise keeps original.
    """
    from rapidfuzz import fuzz
    aligned = []
    for start, end, text in segments:
        best_score = 0
        best_line = text
        for line in desc_lines:
            score = fuzz.partial_ratio(text, line)
            if score > best_score:
                best_score = score
                best_line = line
        aligned.append((start, end, best_line if best_score >= threshold else text))
    return aligned


def download_video(url: str, work_dir: Path, lang: str) -> tuple[Path | None, Path | None]:
    """Download audio + subtitles for a URL.

    Returns (audio_path, vtt_path). Either can be None on failure.
    """
    video_id = _get_video_id(url)
    if not video_id:
        print("  Could not resolve video ID — skipping")
        return None, None

    audio_path = work_dir / f"{video_id}.wav"

    # --- audio ---
    if audio_path.exists():
        print(f"  Audio already cached: {audio_path.name}")
    else:
        print(f"  Downloading audio ({video_id})...")
        ret = subprocess.run([
            *YTDLP, "--no-playlist", "-x", "--audio-format", "wav",
            "--ffmpeg-location", _ffmpeg_location(),
            "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
            "-o", str(work_dir / "%(id)s.%(ext)s"),
            url,
        ])
        if ret.returncode != 0 or not audio_path.exists():
            print("  Audio download failed")
            return None, None

    # --- subtitles (manual preferred, auto fallback) ---
    existing = list(work_dir.glob(f"{video_id}*.vtt"))
    if existing:
        print(f"  Subtitles already cached: {existing[0].name}")
        return audio_path, existing[0]

    print(f"  Downloading subtitles ({video_id})...")
    for flag in ["--write-sub", "--write-auto-sub"]:
        subprocess.run([
            *YTDLP, "--no-playlist", flag,
            "--sub-lang", f"{lang},{lang}-*",
            "--convert-subs", "vtt",
            "--ffmpeg-location", _ffmpeg_location(),
            "--skip-download",
            "-o", str(work_dir / "%(id)s.%(ext)s"),
            url,
        ])
        found = list(work_dir.glob(f"{video_id}*.vtt"))
        if found:
            return audio_path, found[0]

    print(f"  No subtitles found in language '{lang}'")
    return audio_path, None


# ---------------------------------------------------------------------------
# Audio segmentation
# ---------------------------------------------------------------------------

def segment_audio(
    audio_path: Path,
    segments: list[tuple[int, int, str]],
    seg_dir: Path,
    video_id: str,
    padding_ms: int,
    min_dur_ms: int,
    max_dur_ms: int,
) -> list[tuple[str, str]]:
    """Slice audio file according to subtitle timestamps.

    Returns list of (wav_path, transcript_text).
    Uses soundfile + numpy — no pydub dependency.
    """
    import soundfile as sf
    import numpy as np

    audio, sr = sf.read(str(audio_path), dtype="float32")
    total_samples = len(audio)
    rows: list[tuple[str, str]] = []

    for i, (start_ms, end_ms, text) in enumerate(segments):
        dur = end_ms - start_ms
        if dur < min_dur_ms or dur > max_dur_ms:
            continue
        s = max(0, int((start_ms - padding_ms) * sr / 1000))
        e = min(total_samples, int((end_ms + padding_ms) * sr / 1000))
        clip = audio[s:e]
        out_path = seg_dir / f"{video_id}_seg_{i:04d}.wav"
        sf.write(str(out_path), clip, sr)
        rows.append((str(out_path), text))

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    urls = read_urls(args.urls)
    print(f"Found {len(urls)} URL(s) to process\n")

    output_dir = Path(args.output_dir)
    work_dir = output_dir / "_downloads"
    work_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[tuple[str, str]] = []

    for idx, url in enumerate(urls, 1):
        print(f"[{idx}/{len(urls)}] {url}")

        audio_path, vtt_path = download_video(url, work_dir, args.lang)
        if audio_path is None:
            print("  Skipped — audio unavailable\n")
            continue
        if vtt_path is None:
            print("  Skipped — no subtitles\n")
            continue

        segments = parse_vtt(str(vtt_path))
        segments = deduplicate_segments(segments)
        print(f"  Parsed {len(segments)} subtitle segments")

        # Try to improve transcript quality using description lyrics
        print(f"  Fetching description for lyric alignment...")
        description = download_description(url)
        desc_lines = extract_devanagari_lines(description)
        if len(desc_lines) >= 5:
            print(f"  Found {len(desc_lines)} Hindi lines in description — aligning transcripts")
            segments = align_with_description(segments, desc_lines)
        else:
            print(f"  No usable Hindi lyrics in description — keeping subtitle text")

        video_id = audio_path.stem
        seg_dir = output_dir / video_id
        seg_dir.mkdir(exist_ok=True)

        rows = segment_audio(
            audio_path, segments, seg_dir, video_id,
            padding_ms=args.padding_ms,
            min_dur_ms=args.min_duration_ms,
            max_dur_ms=args.max_duration_ms,
        )
        print(f"  Created {len(rows)} audio segments\n")
        all_rows.extend(rows)

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "transcript"])
        writer.writerows(all_rows)

    print(f"{'─'*50}")
    print(f"Total segments : {len(all_rows)}")
    print(f"Training CSV   : {args.out_csv}")
    print(f"\nNext step:")
    print(f"  python train_indicwhisper.py \\")
    print(f"    --train_csv {args.out_csv} \\")
    print(f"    --output_dir outputs/indicwhisper_bhajan")


if __name__ == "__main__":
    main()

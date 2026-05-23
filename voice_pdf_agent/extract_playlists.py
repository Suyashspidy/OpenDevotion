#!/usr/bin/env python3
"""Extract bhajan URLs from playlists + YouTube search to reach ~200 unique videos."""
import subprocess
import sys
import json
from pathlib import Path

PLAYLISTS = [
    "PLOOUvAArsUcaYbO5v8ZhDzGTUGqwARZ3x",
    "PLOOUvAArsUcb5MN1TeWghTphSNk7P8lqZ",
    "PLrblCMBhoy4x5zbRfHcXhJg5Ka9BT_Frw",
    "RDGbWSX749dAk",
    "RDUG-kcoLYo70",
    "RDWH9y4vVjSyg",
    "RDbsCv077CXQ0",
]

SEARCH_QUERIES = [
    "हिंदी भजन संग्रह",
    "Sanskrit bhajan Hindi",
    "Anuradha Paudwal bhajan",
    "Anup Jalota bhajan",
    "Vinod Agrawal bhajan",
    "Swami Mukundananda bhajan",
    "Jaya Kishori bhajan",
    "Hare Krishna bhajan Hindi",
    "Ram bhajan Hindi devotional",
    "Hanuman bhajan Hindi",
    "Shiv bhajan Hindi",
    "bhajan kirtan Hindi",
]

TARGET = 200


def _extract_video_id(url: str) -> str | None:
    """Normalise a YouTube URL to its bare video ID."""
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    if "v" in params:
        return params["v"][0]
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/").split("/")[0]
    return None


def _canonical(url: str) -> str:
    """Return canonical watch URL from any YouTube URL."""
    vid = _extract_video_id(url)
    return f"https://www.youtube.com/watch?v={vid}" if vid else url


def load_existing_urls() -> list[str]:
    """Load and deduplicate URLs from both urls.txt and expanded_urls.txt."""
    seen: set[str] = set()
    result: list[str] = []
    for fname in ("urls.txt", "expanded_urls.txt"):
        path = Path(fname)
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                canonical = _canonical(line)
                if canonical not in seen:
                    seen.add(canonical)
                    result.append(canonical)
    return result


def extract_playlist(playlist_id: str, max_videos: int = 50) -> list[str]:
    print(f"  Playlist {playlist_id}...")
    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    cmd = [sys.executable, "-m", "yt_dlp", "--flat-playlist", "-j", "-I", f"1:{max_videos}", url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        urls = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                vid_id = json.loads(line).get("id")
                if vid_id:
                    urls.append(f"https://www.youtube.com/watch?v={vid_id}")
            except Exception:
                pass
        print(f"    {len(urls)} videos")
        return urls
    except subprocess.TimeoutExpired:
        print("    TIMEOUT")
        return []
    except Exception as e:
        print(f"    ERROR: {e}")
        return []


def search_youtube(query: str, max_results: int = 30) -> list[str]:
    print(f"  Search: '{query}'...")
    cmd = [sys.executable, "-m", "yt_dlp", "--flat-playlist", "-j", f"ytsearch{max_results}:{query}"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        urls = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                vid_id = json.loads(line).get("id")
                if vid_id:
                    urls.append(f"https://www.youtube.com/watch?v={vid_id}")
            except Exception:
                pass
        print(f"    {len(urls)} results")
        return urls
    except subprocess.TimeoutExpired:
        print("    TIMEOUT")
        return []
    except Exception as e:
        print(f"    ERROR: {e}")
        return []


def main():
    all_urls = load_existing_urls()
    seen = set(all_urls)
    print(f"Loaded {len(all_urls)} existing unique URLs (urls.txt + expanded_urls.txt)")

    print("\n[Playlists]")
    for pid in PLAYLISTS:
        if len(all_urls) >= TARGET:
            break
        for url in extract_playlist(pid):
            if url not in seen:
                all_urls.append(url)
                seen.add(url)

    print(f"\n[Search] — {max(0, TARGET - len(all_urls))} more needed")
    for query in SEARCH_QUERIES:
        if len(all_urls) >= TARGET:
            break
        needed = TARGET - len(all_urls)
        for url in search_youtube(query, max_results=min(needed + 10, 50)):
            if url not in seen:
                all_urls.append(url)
                seen.add(url)
            if len(all_urls) >= TARGET:
                break

    print(f"\nTotal unique URLs: {len(all_urls)}")

    out_path = Path("expanded_urls.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for url in all_urls:
            f.write(url + "\n")
    print(f"Saved to: {out_path}")

    print(f"""
Next steps:
  1. Download + segment all {len(all_urls)} videos (cached ones are skipped automatically):
       python prepare_training_data.py --urls expanded_urls.txt --output_dir data/bhajan_segments --out_csv bhajan_train_expanded.csv

  2. Retrain on the full dataset:
       python train_indicwhisper.py --train_csv bhajan_train_expanded.csv --output_dir outputs/indicwhisper_bhajan_v2
""")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Expand bhajan URL collection from existing seeds."""
import sys
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# Existing 10 seed URLs from urls.txt
SEED_URLS = [
    "https://www.youtube.com/watch?v=UG-kcoLYo70&list=RDUG-kcoLYo70&start_radio=1",
    "https://www.youtube.com/watch?v=WH9y4vVjSyg&list=RDWH9y4vVjSyg&start_radio=1",
    "https://www.youtube.com/watch?v=lvpInxM35qQ&list=PLOOUvAArsUcaYbO5v8ZhDzGTUGqwARZ3x",
    "https://www.youtube.com/watch?v=bsCv077CXQ0&list=PLOOUvAArsUcb5MN1TeWghTphSNk7P8lqZ",
    "https://www.youtube.com/watch?v=WTHyqsBHUHU&list=RDbsCv077CXQ0&index=3",
    "https://www.youtube.com/watch?v=Mstbp2TzErY&list=RDbsCv077CXQ0&index=7",
    "https://www.youtube.com/watch?v=YVjFCNPBnIM&list=RDbsCv077CXQ0&index=23",
    "https://www.youtube.com/watch?v=RFA4ViWD9vE&list=PLrblCMBhoy4x5zbRfHcXhJg5Ka9BT_Frw",
    "https://www.youtube.com/watch?v=9oHwBCXU_fM&list=PLrblCMBhoy4x5zbRfHcXhJg5Ka9BT_Frw&index=4",
    "https://www.youtube.com/watch?v=GbWSX749dAk&list=RDGbWSX749dAk&start_radio=1",
]

def extract_playlist_ids(urls):
    """Extract unique playlist IDs from URLs."""
    playlists = set()
    for url in urls:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if "list" in params:
            playlists.add(params["list"][0])
    return sorted(playlists)

def extract_channel_from_url(url):
    """Extract channel if visible in URL."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    if "channel" in params:
        return params["channel"][0]
    return None

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("BHAJAN URL EXPANSION ANALYSIS")
    print("=" * 70)

    print(f"\n[OK] Seed URLs: {len(SEED_URLS)}")

    playlists = extract_playlist_ids(SEED_URLS)
    print(f"\n[PLAYLISTS] Unique Playlists Found: {len(playlists)}")
    for i, pid in enumerate(playlists, 1):
        print(f"   {i}. https://www.youtube.com/playlist?list={pid}")

    print("\n" + "=" * 70)
    print("NEXT STEPS TO EXPAND TO 50-100+ URLs:")
    print("=" * 70)
    print("""
1. EXTRACT PLAYLISTS (automated via yt-dlp):
   For each playlist above, run:
   yt-dlp --flat-playlist -j "https://www.youtube.com/playlist?list=PLAYLIST_ID" | jq -r '.id' | sed "s/^/https:\\/\\/www.youtube.com\\/watch?v=/" > playlist_urls.txt

2. MANUAL SEARCH (targeted):
   - Search "Sanskrit भजन" (Sanskrit Bhajans)
   - Search "हिंदी भजन संग्रह" (Hindi Bhajan Collections)
   - Search channels: Anuradha Paudwal, Times Music Bhajan, Sarangi Music

3. LANGUAGE FILTER:
   - Keep Hindi + Sanskrit focused
   - Skip Marathi/Bengali for now
   - Prioritize 2-15 min duration videos
   - Skip instrumental-only versions

4. EXPECTED OUTCOME:
   - 50-100 URLs from playlists
   - 5000-10000 training segments
   - Better language coverage
   - Diverse vocal styles + artists
""")

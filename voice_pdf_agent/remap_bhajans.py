"""
Parse the PDF's own index (pages 14-19) to get authoritative bhajan names
and page boundaries, then remap bhajan_ids in index_meta.npy and rewrite
bhajans.json — no OCR rebuild needed.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import fitz
import numpy as np


# ── Section/sub-section headers that appear without page numbers ──────────────

_HEADERS = {
    "INDEX OF KEERTANS", "NAAM SANKEERTAN", "BHAJAN", "ANYA BHAJAN",
    "PREM RAS MADIRA", "SADHAN BHAKTI TATVA", "KAVVALI",
    "SADGURU MADHURI", "SIDDHANT MADHURI", "DAINYA MADHURI",
    "SHREE KRISHNA BAAL-LEELA MADHURI", "SHREE RADHA BAAL-LEELA MADHURI",
    "SHREE KRISHNA MADHURI", "SHREE RADHA MADHURI", "YUGAL MADHURI",
    "LEELA MADHURI", "NIKUNJ MADHURI", "MILAN MADHURI", "VIRAH MADHURI",
    "PRAKEERNA MADHURI",
}


def _is_header(line: str) -> bool:
    norm = line.upper().strip(".,!? ")
    if norm in _HEADERS:
        return True
    if "MADHURI" in norm or "BAAL-LEELA" in norm:
        return True
    return False


def parse_pdf_index(pdf_path: str, index_pages=range(13, 19)) -> list[tuple[str, int]]:
    """Return list of (bhajan_name, 1-based page number) from the PDF index."""
    doc = fitz.open(pdf_path)
    entries: list[tuple[str, int]] = []
    name_buf: list[str] = []

    for pi in index_pages:
        if pi >= len(doc):
            break
        for raw_line in doc[pi].get_text().split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            # Skip the index page numbers themselves (14-19)
            if re.match(r"^1[4-9]$", line):
                name_buf.clear()
                continue

            # Skip section/sub-section headers
            if _is_header(line):
                name_buf.clear()
                continue

            # Case 1: "Name  page_number" on same line (1+ spaces before number)
            m = re.match(r"^(.+?)\s+(\d{2,3})\s*$", line)
            if m:
                page_num = int(m.group(2))
                if 30 <= page_num <= 210:
                    name_part = m.group(1).strip()
                    if name_part:
                        name_buf.append(name_part)
                    if name_buf:
                        entries.append((" ".join(name_buf), page_num))
                    name_buf.clear()
                    continue

            # Case 2: standalone page number on its own line
            m2 = re.match(r"^(\d{2,3})$", line)
            if m2:
                page_num = int(m2.group(1))
                if 30 <= page_num <= 210 and name_buf:
                    entries.append((" ".join(name_buf), page_num))
                    name_buf.clear()
                continue

            # Otherwise: accumulate as (part of) a bhajan name
            name_buf.append(line)

    return entries


_SECTION_KEYWORDS = [
    'आरती', 'वन्दना', 'वंदना', 'प्रार्थना', 'भोग',
    'स्तुति', 'स्तवन', 'मंगलाचरण', 'अर्चना',
]

# Sections whose first OCR'd line is already a lyric/shloka (no title page),
# so auto-detection can't find them. Format: (title, 1-based page number).
_MANUAL_PRE_KEERTAN: list[tuple[str, int]] = [
    ("वन्दना",   30),
    ("भोग आरती", 35),
]


def _detect_pre_keertan_sections(
    chunks: list[dict],
    start_0: int,
    end_0: int,
) -> list[tuple[str, int]]:
    """Detect section starts in pre-keertan pages using OCR'd chunk text.

    Uses the already-OCR'd text stored in index_meta.npy (raw PDF extraction
    is useless for these pages — Devanagari needs OCR). Groups chunks by page,
    then checks whether the first Devanagari line contains a known section
    keyword (आरती, वन्दना, प्रार्थना, etc.) and is short enough to be a title.
    This avoids the false positives that the generic length-ratio heuristic
    produces on lyric continuation pages.

    Returns list of (title, 1-based-page).
    """
    from collections import defaultdict
    page_lines: dict[int, list[str]] = defaultdict(list)
    for c in chunks:
        p = c.get("page", -1)
        if start_0 <= p < end_0:
            text = c.get("text", "").strip()
            if text:
                page_lines[p].append(text)

    if not page_lines:
        return []

    entries: list[tuple[str, int]] = []
    for page_0 in sorted(page_lines):
        lines = page_lines[page_0]
        deva_lines = [l for l in lines if _is_mostly_devanagari(l)]
        if not deva_lines:
            continue
        first = deva_lines[0]
        # A section title page starts with a short line that names the section.
        # Continuation and meaning pages start with a lyric or prose line that
        # won't contain these keywords.
        if len(first) <= 40 and any(kw in first for kw in _SECTION_KEYWORDS):
            entries.append((first[:80], page_0 + 1))  # 1-based

    return entries


def _is_mostly_devanagari(text: str, threshold: float = 0.4) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    devanagari = sum(1 for c in letters if 'ऀ' <= c <= 'ॿ')
    return devanagari / len(letters) >= threshold


def remap(pdf_path: str) -> None:
    base = Path(pdf_path).stem
    pdf_dir = Path(pdf_path).parent
    meta_path = pdf_dir / f"{base}.index_meta.npy"
    bhajans_path = pdf_dir / f"{base}.bhajans.json"

    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Run build_index.py first.")
        sys.exit(1)

    # ── Parse PDF index ───────────────────────────────────────────────────────
    entries = parse_pdf_index(pdf_path)
    print(f"Parsed {len(entries)} bhajans from PDF index.")
    for name, pg in entries[:5]:
        print(f"  page {pg}: {name}")
    print("  ...")

    # ── Load existing index_meta (needed for pre-keertan detection) ──────────
    raw_meta = np.load(str(meta_path), allow_pickle=True)
    chunks = [m.item() if hasattr(m, "item") else m for m in raw_meta]
    print(f"Loaded {len(chunks)} chunks from index_meta.")

    # ── Detect pre-keertan sections (aarti, vandana, prarthana, etc.) ─────────
    # The keertan index spans pages 14–19 (1-based), so pre-keertan content
    # starts at page 20 (0-indexed: 19). Scan up to the first keertan page.
    # Use OCR'd chunk text from index_meta — raw PDF extraction gives garbage.
    INDEX_END_0 = 19  # first page after the keertan index (0-indexed)
    first_keertan_0 = (min(pg - 1 for _, pg in entries) if entries else INDEX_END_0)
    pre_keertan = _detect_pre_keertan_sections(chunks, INDEX_END_0, first_keertan_0)
    all_pre = pre_keertan + [e for e in _MANUAL_PRE_KEERTAN
                             if not any(abs(e[1] - pg) <= 1 for _, pg in pre_keertan)]
    if all_pre:
        print(f"Pre-keertan sections ({len(pre_keertan)} auto + "
              f"{len(all_pre) - len(pre_keertan)} manual):")
        for name, pg in sorted(all_pre, key=lambda x: x[1]):
            print(f"  page {pg}: {name}")
        entries = all_pre + entries
    else:
        print("No pre-keertan sections detected (pages 20–first keertan).")

    # ── Build boundary list: sorted by 0-indexed start page ──────────────────
    # Multiple bhajans can share the same start page — keep all
    # entries sorted by page, assign sequential bhajan_ids
    entries.sort(key=lambda x: x[1])
    bhajan_starts_1based = [pg for _, pg in entries]
    bhajan_names = [name for name, _ in entries]

    # Convert to 0-based page indices
    bhajan_starts_0based = [pg - 1 for pg in bhajan_starts_1based]

    # ── Remap bhajan_id for each chunk ────────────────────────────────────────
    # For a chunk on page p, its bhajan is the last entry whose start_page <= p
    for chunk in chunks:
        p = chunk.get("page", 0)
        bid = 0
        for i, start in enumerate(bhajan_starts_0based):
            if start <= p:
                bid = i
            else:
                break
        chunk["bhajan_id"] = bid

    # ── Alias chunks for co-page bhajans ─────────────────────────────────────
    # When N bhajans share the same start page, the loop above gives all chunks
    # on that page to the LAST bhajan; the earlier ones end up with zero chunks
    # and are silently dropped from any setlist. Fix: for each co-page group,
    # find which bhajans have no chunks and copy the winner's page chunks to
    # them (same text, different bhajan_id). FAISS vectors aren't duplicated so
    # semantic search won't hit these aliases, but fuzzy search will work.
    from collections import defaultdict
    page_to_chunk_indices: dict[int, list[int]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        page_to_chunk_indices[chunk.get("page", 0)].append(idx)

    bid_has_chunks: set[int] = {chunk["bhajan_id"] for chunk in chunks}

    # Group bhajan_ids by shared start page
    start_page_groups: dict[int, list[int]] = defaultdict(list)
    for bid, sp in enumerate(bhajan_starts_0based):
        start_page_groups[sp].append(bid)

    alias_chunks: list[dict] = []
    for sp, bids in start_page_groups.items():
        if len(bids) < 2:
            continue
        empty_bids = [b for b in bids if b not in bid_has_chunks]
        if not empty_bids:
            continue
        # Collect source chunks: pages from this start up to the next group
        next_sp = min((s for s in start_page_groups if s > sp), default=sp + 1)
        source_chunks = [
            chunks[idx]
            for p in range(sp, next_sp)
            for idx in page_to_chunk_indices.get(p, [])
        ]
        for empty_bid in empty_bids:
            for src in source_chunks:
                alias = dict(src)
                alias["bhajan_id"] = empty_bid
                alias_chunks.append(alias)

    if alias_chunks:
        n_fixed = len({c["bhajan_id"] for c in alias_chunks})
        print(f"Added {len(alias_chunks)} alias chunks for {n_fixed} co-page bhajan(s) with no text.")
        chunks = chunks + alias_chunks

    # ── Save updated index_meta ───────────────────────────────────────────────
    np.save(str(meta_path), np.array(chunks, dtype=object))
    print(f"Saved updated index_meta ->{meta_path}")

    # ── Save new bhajans.json with proper English names ───────────────────────
    bhajans_data = [
        {
            "bhajan_id": i,
            "title": bhajan_names[i],
            "start_page": bhajan_starts_0based[i],
        }
        for i in range(len(entries))
    ]
    with open(bhajans_path, "w", encoding="utf-8") as f:
        json.dump(bhajans_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(bhajans_data)} bhajans ->{bhajans_path}")

    # Quick sanity check
    print("\nFirst 10 bhajans in dropdown:")
    for bj in bhajans_data[:10]:
        print(f"  [{bj['bhajan_id']}] page {bj['start_page']+1}: {bj['title']}")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "Sankirtan Madhuri.pdf"
    remap(pdf)

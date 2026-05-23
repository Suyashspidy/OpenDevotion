# Bhajan Live Reader — Devotee Quick-Start Guide

*One page. Read once before the keertan.*

---

## Before the session

1. **Open the app** — double-click the launcher or run `python app.py` from the `voice_pdf_agent` folder.
2. **Load the songbook** — click **Browse…** next to PDF 1 and select the bhajan PDF.
3. **Detect bhajans** — click **Detect Bhajans ▼**. Wait a few seconds for the list to load.
4. **Set the microphone** — pick the microphone closest to the harmonium or lead singer from the dropdown. Click **Refresh** if nothing appears.
5. **Set the language** — choose the script the PDF is in (e.g., **हिन्दी** for Hindi, **संस्कृत** for Sanskrit).

---

## Optional: Build a setlist

If you know today's program in advance:

- Use the **🔍 Search** buttons next to slots 1–5 to pre-load the bhajans in order.
- Tick **Enforce performance order** if you want the tracker to follow the list strictly (it will never go backwards).
- Leave slots blank to let the app follow freely without a setlist.

---

## Starting the session

Click **▶ Start Session**.

A second window opens showing the PDF page. The app listens continuously — **you do not need to press anything during the keertan**. The page turns automatically as the bhajan progresses.

---

## During the keertan

| Situation | What to do |
|---|---|
| Page did not turn automatically | Sing the next line clearly into the mic; it will catch up |
| Wrong bhajan detected | Click anywhere on the correct page to jump manually |
| Too much background noise | Move the mic closer to the singer or harmonium |
| App seems frozen | Check the status bar at the bottom of the launcher window |

---

## Ending the session

Close the lyric window or press **Ctrl + Q**. The setlist is saved automatically for next time.

---

## Tips for best results

- **Mic placement** matters most. A small lapel mic near the harmonium works better than a room mic.
- The app works best with **IndicWhisper** selected as the ASR model.
- Start speaking/singing **2–3 seconds** after clicking Start to let the model load.
- Keep laptop **plugged in** — the GPU is used for transcription and needs full power.

---

*For setup help or issues, contact the person who installed the system.*

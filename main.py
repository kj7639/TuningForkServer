import os
import re
import subprocess
import tempfile
import logging
import httpx
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
import pathlib as _pathlib
import music21
from music21 import converter

_MODEL_DIR = _pathlib.Path(str(ICASSP_2022_MODEL_PATH)).parent
ONNX_MODEL_PATH = str(_MODEL_DIR / "nmp.onnx")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tuning Fork API",
    description="Converts a song to sheet music (MusicXML) or chord-annotated lyrics",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SongRequest(BaseModel):
    song_name: str
    artist: str


# ── Shared: audio download ────────────────────────────────────────────────────

def search_and_download_audio(song_name: str, artist: str, output_dir: str) -> str:
    """Download the best audio match from YouTube as MP3 via yt-dlp CLI."""
    query = f"ytsearch1:{artist} - {song_name}"
    mp3_path = os.path.join(output_dir, "audio.mp3")
    cmd = [
        "yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "192K",
        "--no-playlist", "--match-filter", "duration < 600", "--retries", "3",
        "-o", mp3_path, query,
    ]
    logger.info(f"Running yt-dlp for: {artist} - {song_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {result.stderr.strip()}")
    if not os.path.exists(mp3_path):
        raise HTTPException(status_code=500, detail="MP3 file not found after download.")
    return mp3_path


# ── Shared: MIDI conversion ───────────────────────────────────────────────────

def convert_audio_to_midi(mp3_path: str, output_dir: str) -> str:
    """Run Basic Pitch inference and return path to the generated MIDI file."""
    predict_and_save(
        audio_path_list=[mp3_path],
        output_directory=output_dir,
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ONNX_MODEL_PATH,
    )
    stem = Path(mp3_path).stem
    midi_path = os.path.join(output_dir, f"{stem}_basic_pitch.mid")
    if not os.path.exists(midi_path):
        mid_files = list(Path(output_dir).glob("*.mid"))
        if not mid_files:
            raise HTTPException(status_code=500, detail="No MIDI file produced.")
        midi_path = str(mid_files[0])
    logger.info(f"MIDI written to: {midi_path}")
    return midi_path


# ── Sheet music: MIDI → MusicXML ─────────────────────────────────────────────

def convert_midi_to_musicxml(midi_path: str, output_dir: str) -> str:
    """Convert a MIDI file to a MusicXML string via music21."""
    logger.info(f"Converting MIDI to MusicXML: {midi_path}")
    score = converter.parse(midi_path)
    score = score.makeNotation()
    xml_path = os.path.join(output_dir, "score.musicxml")
    score.write("musicxml", fp=xml_path)
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()
    logger.info("MusicXML conversion complete")
    return xml_content


# ── Chords: extraction ────────────────────────────────────────────────────────

def _chord_label(root: str, quality: str) -> str:
    q_map = {
        "major": "", "minor": "m", "diminished": "dim", "augmented": "aug",
        "dominant-seventh": "7", "major-seventh": "maj7", "minor-seventh": "m7",
        "half-diminished": "m7b5", "diminished-seventh": "dim7",
        "suspended-fourth": "sus4", "suspended-second": "sus2",
    }
    return f"{root}{q_map.get(quality, '')}"


def _offset_to_seconds(offset, score) -> float:
    try:
        mm = score.flatten().getElementsByClass(music21.tempo.MetronomeMark)
        bpm = mm[0].number if mm else 120.0
    except Exception:
        bpm = 120.0
    return float(offset) * (60.0 / bpm)


def extract_chords(midi_path: str) -> list[dict]:
    logger.info("Extracting chords from MIDI...")
    score = converter.parse(midi_path)
    chordified = score.chordify()
    chords_out = []
    last_label = None
    for c in chordified.flatten().getElementsByClass(music21.chord.Chord):
        if len(c.pitches) < 2:
            continue
        try:
            root = c.root().name
            quality = c.quality
        except Exception:
            continue
        label = _chord_label(root, quality)
        if label == last_label:
            continue
        last_label = label
        chords_out.append({
            "chord": label,
            "quality": quality,
            "root": root,
            "timestamp": round(_offset_to_seconds(c.offset, score), 2),
        })
    logger.info(f"Extracted {len(chords_out)} unique chords")
    return chords_out


# ── Chords: lyrics fetch + alignment ─────────────────────────────────────────

def fetch_synced_lyrics(song_name: str, artist: str) -> list[dict] | None:
    """Fetch synced LRC lyrics from lrclib.net. Returns None if unavailable."""
    try:
        resp = httpx.get(
            "https://lrclib.net/api/search",
            params={"track_name": song_name, "artist_name": artist},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            logger.warning("lrclib returned no results")
            return None
        synced_lrc = next((r["syncedLyrics"] for r in results if r.get("syncedLyrics")), None)
        if not synced_lrc:
            logger.warning("No synced lyrics found in lrclib results")
            return None
        return parse_lrc(synced_lrc)
    except Exception as e:
        logger.warning(f"Failed to fetch lyrics: {e}")
        return None


def parse_lrc(lrc: str) -> list[dict]:
    """Parse LRC timestamp format into list of { time, text } dicts."""
    pattern = re.compile(r"\[(\d+):(\d+\.\d+)\](.*)")
    lines = []
    for raw_line in lrc.splitlines():
        m = pattern.match(raw_line.strip())
        if not m:
            continue
        text = m.group(3).strip()
        if not text:
            continue
        lines.append({
            "time": round(int(m.group(1)) * 60 + float(m.group(2)), 2),
            "text": text,
        })
    return sorted(lines, key=lambda x: x["time"])


def align_chords_to_lyrics(lyric_lines: list[dict], chords: list[dict]) -> list[dict]:
    """Attach chords to the lyric line they fall within by timestamp."""
    enriched = []
    for i, line in enumerate(lyric_lines):
        line_start = line["time"]
        line_end = lyric_lines[i + 1]["time"] if i + 1 < len(lyric_lines) else float("inf")
        seen: set[str] = set()
        unique_chords: list[str] = []
        for c in chords:
            if line_start <= c["timestamp"] < line_end and c["chord"] not in seen:
                seen.add(c["chord"])
                unique_chords.append(c["chord"])
        enriched.append({"time": line["time"], "text": line["text"], "chords": unique_chords})
    return enriched


# ── Shared: cleanup ───────────────────────────────────────────────────────────

def _delete_dir(path: str) -> None:
    import shutil
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


# ── Endpoint 1: sheet music ───────────────────────────────────────────────────

@app.post("/sheet-music", summary="Convert a song to MusicXML sheet music")
async def get_sheet_music(request: SongRequest):
    """
    Downloads audio, runs Basic Pitch, converts MIDI → MusicXML via music21.
    Returns JSON: { song_name, artist, musicxml }
    """
    work_dir = tempfile.mkdtemp(prefix="song2sheet_")
    try:
        mp3_path = search_and_download_audio(request.song_name, request.artist, work_dir)
        midi_path = convert_audio_to_midi(mp3_path, work_dir)
        os.remove(mp3_path)
        musicxml = convert_midi_to_musicxml(midi_path, work_dir)
        return JSONResponse(content={
            "song_name": request.song_name,
            "artist": request.artist,
            "musicxml": musicxml,
        })
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /sheet-music")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _delete_dir(work_dir)


# ── Endpoint 2: chords + lyrics ───────────────────────────────────────────────

@app.post("/chords", summary="Convert a song to chord-annotated lyrics")
async def get_chords(request: SongRequest):
    """
    Downloads audio, runs Basic Pitch, extracts chords, fetches synced lyrics
    from lrclib and aligns chords to lyric lines.
    Returns JSON: { song_name, artist, has_lyrics, lines, all_chords }
    """
    work_dir = tempfile.mkdtemp(prefix="song2chords_")
    try:
        mp3_path = search_and_download_audio(request.song_name, request.artist, work_dir)
        midi_path = convert_audio_to_midi(mp3_path, work_dir)
        os.remove(mp3_path)
        chords = extract_chords(midi_path)
        lyric_lines = fetch_synced_lyrics(request.song_name, request.artist)
        if lyric_lines:
            lines = align_chords_to_lyrics(lyric_lines, chords)
            has_lyrics = True
        else:
            lines = [{"time": c["timestamp"], "text": "", "chords": [c["chord"]]} for c in chords]
            has_lyrics = False
        return JSONResponse(content={
            "song_name": request.song_name,
            "artist": request.artist,
            "has_lyrics": has_lyrics,
            "lines": lines,
            "all_chords": list({c["chord"] for c in chords}),
        })
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /chords")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _delete_dir(work_dir)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
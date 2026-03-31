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
from music21 import converter, tempo, meter, key, note, chord, stream

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


# ── Post-processing ───────────────────────────────────────────────────────────

# Minimum note duration to keep — anything shorter is likely transcription noise.
# music21 uses quarter lengths: 0.125 = 32nd note, 0.25 = 16th note
MIN_NOTE_DURATION = 0.125

# Quantisation grid in quarter lengths.
# 0.25 = 16th note grid (good balance of accuracy vs cleanliness)
QUANTISE_GRID = 0.25


def smooth_tempo(score: stream.Score) -> float:
    """
    Extract all tempo markings from the score, compute a weighted median BPM,
    and replace all MetronomeMark objects with a single stable tempo.
    Returns the chosen BPM.
    """
    marks = score.flatten().getElementsByClass(tempo.MetronomeMark)
    bpms = [m.number for m in marks if m.number and 40 <= m.number <= 240]

    if not bpms:
        chosen_bpm = 120.0
    else:
        bpms.sort()
        chosen_bpm = float(bpms[len(bpms) // 2])  # median

    # Remove all existing tempo marks and insert one at the top
    for part in score.parts:
        for m in part.flatten().getElementsByClass(tempo.MetronomeMark):
            m.activeSite.remove(m)

    score.parts[0].measure(0 if score.parts[0].measure(0) else 1).insert(
        0, tempo.MetronomeMark(number=chosen_bpm)
    )
    logger.info(f"Tempo smoothed to {chosen_bpm} BPM")
    return chosen_bpm


def detect_and_set_key(score: stream.Score) -> key.Key:
    """
    Analyse the score to detect its key, insert a KeySignature at the start,
    and respell notes enharmonically to match the key.
    """
    detected = score.analyze("key")
    logger.info(f"Detected key: {detected}")

    ks = detected.asKey() if hasattr(detected, "asKey") else key.Key(detected.tonic.name, detected.mode)

    for part in score.parts:
        # Insert key signature at the very beginning
        measures = part.getElementsByClass(stream.Measure)
        if measures:
            measures[0].insert(0, ks)

        # Respell all notes to match the key (e.g. Gb → F# in G major)
        for n in part.flatten().getElementsByClass(note.Note):
            try:
                n.pitch.simplifyEnharmonic(inPlace=True, mostCommon=True)
            except Exception:
                pass

    return detected


def infer_time_signature(score: stream.Score) -> str:
    """
    Check if the score already has a sensible time signature; if not,
    attempt to infer one from the note density, defaulting to 4/4.
    Returns the time signature string used.
    """
    existing = score.flatten().getElementsByClass(meter.TimeSignature)
    if existing:
        ts_str = existing[0].ratioString
        logger.info(f"Keeping existing time signature: {ts_str}")
        return ts_str

    # Simple heuristic: count notes per bar and pick the most common grouping
    ts_str = "4/4"
    try:
        notes_per_measure = []
        for part in score.parts:
            for m in part.getElementsByClass(stream.Measure):
                n_count = len(m.flatten().getElementsByClass(note.Note))
                if n_count > 0:
                    notes_per_measure.append(n_count)
        if notes_per_measure:
            avg = sum(notes_per_measure) / len(notes_per_measure)
            if avg <= 3:
                ts_str = "3/4"
            elif avg >= 6:
                ts_str = "6/8"
            else:
                ts_str = "4/4"
    except Exception:
        pass

    ts = meter.TimeSignature(ts_str)
    for part in score.parts:
        measures = part.getElementsByClass(stream.Measure)
        if measures:
            measures[0].insert(0, ts)

    logger.info(f"Set time signature: {ts_str}")
    return ts_str


def filter_short_notes(score: stream.Score, min_duration: float = MIN_NOTE_DURATION) -> stream.Score:
    """
    Remove notes and chords shorter than min_duration quarter lengths.
    These are almost always transcription artefacts rather than real notes.
    """
    removed = 0
    for part in score.parts:
        for n in part.flatten().getElementsByClass((note.Note, chord.Chord)):
            if n.duration.quarterLength < min_duration:
                try:
                    n.activeSite.remove(n)
                    removed += 1
                except Exception:
                    pass
    logger.info(f"Removed {removed} short notes (< {min_duration} QL)")
    return score


def quantise_score(score: stream.Score, grid: float = QUANTISE_GRID) -> stream.Score:
    """
    Snap all note onsets and durations to the nearest grid value.
    This makes the notation significantly cleaner in the rendered score.
    """
    quarter_length_divisors = [1.0 / grid]
    try:
        score = score.quantize(
            quarterLengthDivisors=quarter_length_divisors,
            processOffsets=True,
            processDurations=True,
            inPlace=False,
        )
        logger.info(f"Score quantised to {grid} QL grid")
    except Exception as e:
        logger.warning(f"Quantisation failed, skipping: {e}")
    return score


def merge_short_rests(score: stream.Score, min_rest: float = 0.25) -> stream.Score:
    """
    Remove rests shorter than min_rest quarter lengths to reduce clutter.
    Short rests between notes are often quantisation artefacts.
    """
    removed = 0
    for part in score.parts:
        for r in part.flatten().getElementsByClass(note.Rest):
            if r.duration.quarterLength < min_rest:
                try:
                    r.activeSite.remove(r)
                    removed += 1
                except Exception:
                    pass
    logger.info(f"Removed {removed} short rests")
    return score


def post_process_score(score: stream.Score) -> stream.Score:
    """
    Apply all post-processing steps in order to clean up a raw Basic Pitch score.
    """
    logger.info("Starting score post-processing...")

    # 1. Filter noise notes first so they don't affect analysis steps
    score = filter_short_notes(score)

    # 2. Quantise to a clean rhythmic grid
    score = quantise_score(score)

    # 3. Remove short rests left behind by quantisation
    score = merge_short_rests(score)

    # 4. Smooth erratic tempo markings to a single stable BPM
    try:
        smooth_tempo(score)
    except Exception as e:
        logger.warning(f"Tempo smoothing failed: {e}")

    # 5. Detect key and respell enharmonics
    try:
        detect_and_set_key(score)
    except Exception as e:
        logger.warning(f"Key detection failed: {e}")

    # 6. Infer or verify time signature
    try:
        infer_time_signature(score)
    except Exception as e:
        logger.warning(f"Time signature inference failed: {e}")

    # 7. Final notation cleanup (beam grouping, tie merging, etc.)
    try:
        score = score.makeNotation(inPlace=False)
    except Exception as e:
        logger.warning(f"makeNotation failed: {e}")

    logger.info("Post-processing complete")
    return score


# ── Sheet music: MIDI → MusicXML ─────────────────────────────────────────────

def convert_midi_to_musicxml(midi_path: str, output_dir: str) -> str:
    logger.info(f"Converting MIDI to MusicXML: {midi_path}")
    score = converter.parse(midi_path)
    score = post_process_score(score)
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


def _offset_to_seconds(offset, score: stream.Score) -> float:
    try:
        mm = score.flatten().getElementsByClass(tempo.MetronomeMark)
        bpm = mm[0].number if mm else 120.0
    except Exception:
        bpm = 120.0
    return float(offset) * (60.0 / bpm)


def extract_chords(midi_path: str, min_chord_duration: float = 0.5) -> list[dict]:
    """
    Parse MIDI, apply post-processing, chordify, then extract chords.
    min_chord_duration filters out chords shorter than N quarter lengths
    to remove spurious detections (default: 0.5 = 8th note).
    """
    logger.info("Extracting chords from MIDI...")
    score = converter.parse(midi_path)

    # Apply the same note filtering and quantisation used for sheet music
    # so chord detection benefits from the same cleanup
    score = filter_short_notes(score)
    score = quantise_score(score)

    # Detect key for better root note spelling
    try:
        detect_and_set_key(score)
    except Exception:
        pass

    chordified = score.chordify()
    chords_out = []
    last_label = None

    for c in chordified.flatten().getElementsByClass(chord.Chord):
        # Skip chords that are too short to be intentional
        if c.duration.quarterLength < min_chord_duration:
            continue
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

    logger.info(f"Extracted {len(chords_out)} chords after filtering")
    return chords_out


# ── Chords: lyrics fetch + alignment ─────────────────────────────────────────

def fetch_synced_lyrics(song_name: str, artist: str) -> list[dict] | None:
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
            logger.warning("No synced lyrics found")
            return None
        return parse_lrc(synced_lrc)
    except Exception as e:
        logger.warning(f"Failed to fetch lyrics: {e}")
        return None


def parse_lrc(lrc: str) -> list[dict]:
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
    Downloads audio, runs Basic Pitch, post-processes and converts
    MIDI → MusicXML via music21.
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
    Downloads audio, runs Basic Pitch, extracts cleaned chords, fetches
    synced lyrics from lrclib and aligns chords to lyric lines.
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
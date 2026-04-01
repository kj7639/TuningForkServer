import os
import re
import subprocess
import tempfile
import logging
import hashlib
from collections import OrderedDict

import numpy as np
import librosa
import httpx
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
import pathlib as _pathlib
from music21 import converter, tempo, meter, key, note, chord, stream

_MODEL_DIR = _pathlib.Path(str(ICASSP_2022_MODEL_PATH)).parent
ONNX_MODEL_PATH = str(_MODEL_DIR / "nmp.onnx")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Trim audio to this many seconds before processing.
# Chord structure repeats throughout a song so 90s captures everything
# while cutting download + inference time by ~70%.
AUDIO_TRIM_SECONDS = 90

# ── Caches ────────────────────────────────────────────────────────────────────
# Audio cache: both endpoints share trimmed MP3 bytes so we download once.
_AUDIO_CACHE: OrderedDict[str, bytes] = OrderedDict()
_AUDIO_CACHE_MAX = 20

# MIDI cache: sheet music endpoint caches Basic Pitch output separately.
_MIDI_CACHE: OrderedDict[str, bytes] = OrderedDict()
_MIDI_CACHE_MAX = 20


def _cache_key(song_name: str, artist: str) -> str:
    return hashlib.sha256(f"{song_name.lower()}|{artist.lower()}".encode()).hexdigest()


# ── Chord templates for chroma matching ──────────────────────────────────────
# Each template is a 12-element unit vector over the chromatic pitch classes,
# ordered C C# D D# E F F# G G# A A# B.

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def _build_templates() -> dict[str, np.ndarray]:
    MAJOR = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)  # root, M3, P5
    MINOR = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)  # root, m3, P5
    DOM7  = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=float)  # root, M3, P5, m7
    MAJ7  = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=float)  # root, M3, P5, M7
    MIN7  = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=float)  # root, m3, P5, m7

    templates: dict[str, np.ndarray] = {}
    qualities = [("", MAJOR), ("m", MINOR), ("7", DOM7), ("maj7", MAJ7), ("m7", MIN7)]
    for i, root in enumerate(_NOTE_NAMES):
        for suffix, base in qualities:
            t = np.roll(base, i).astype(float)
            templates[f"{root}{suffix}"] = t / np.linalg.norm(t)
    return templates

_TEMPLATES = _build_templates()

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Tuning Fork API",
    description="Converts a song to sheet music (MusicXML) or chord-annotated lyrics",
    version="6.0.0",
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


# ── Audio: download + cache ───────────────────────────────────────────────────

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

    # Trim to AUDIO_TRIM_SECONDS — reduces inference time dramatically.
    trimmed_path = os.path.join(output_dir, "audio_trimmed.mp3")
    trim = subprocess.run(
        ["ffmpeg", "-i", mp3_path, "-t", str(AUDIO_TRIM_SECONDS), "-y", trimmed_path],
        capture_output=True, text=True,
    )
    if trim.returncode == 0 and os.path.exists(trimmed_path):
        os.replace(trimmed_path, mp3_path)
        logger.info(f"Audio trimmed to {AUDIO_TRIM_SECONDS}s")
    else:
        logger.warning("ffmpeg trim failed, using full audio")

    return mp3_path


def get_audio(song_name: str, artist: str, output_dir: str) -> str:
    """
    Returns path to a trimmed MP3 for the song.
    Cached in memory so both /chords and /sheet-music share one download.
    """
    key = _cache_key(song_name, artist)

    if key in _AUDIO_CACHE:
        logger.info(f"Audio cache hit for: {artist} - {song_name}")
        audio_path = os.path.join(output_dir, "audio.mp3")
        with open(audio_path, "wb") as f:
            f.write(_AUDIO_CACHE[key])
        _AUDIO_CACHE.move_to_end(key)
        return audio_path

    logger.info(f"Audio cache miss for: {artist} - {song_name}")
    audio_path = search_and_download_audio(song_name, artist, output_dir)

    with open(audio_path, "rb") as f:
        _AUDIO_CACHE[key] = f.read()
    if len(_AUDIO_CACHE) > _AUDIO_CACHE_MAX:
        _AUDIO_CACHE.popitem(last=False)
    logger.info(f"Audio cached ({len(_AUDIO_CACHE)}/{_AUDIO_CACHE_MAX} slots)")

    return audio_path


# ── MIDI: Basic Pitch + cache (used only for sheet music) ─────────────────────

def get_midi(song_name: str, artist: str, output_dir: str) -> str:
    """
    Returns path to a MIDI file produced by Basic Pitch.
    Cached separately from audio so /sheet-music skips re-inference.
    """
    key = _cache_key(song_name, artist)

    if key in _MIDI_CACHE:
        logger.info(f"MIDI cache hit for: {artist} - {song_name}")
        midi_path = os.path.join(output_dir, "cached.mid")
        with open(midi_path, "wb") as f:
            f.write(_MIDI_CACHE[key])
        _MIDI_CACHE.move_to_end(key)
        return midi_path

    audio_path = get_audio(song_name, artist, output_dir)
    midi_path = _run_basic_pitch(audio_path, output_dir)

    with open(midi_path, "rb") as f:
        _MIDI_CACHE[key] = f.read()
    if len(_MIDI_CACHE) > _MIDI_CACHE_MAX:
        _MIDI_CACHE.popitem(last=False)
    logger.info(f"MIDI cached ({len(_MIDI_CACHE)}/{_MIDI_CACHE_MAX} slots)")

    return midi_path


def _run_basic_pitch(mp3_path: str, output_dir: str) -> str:
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


# ── Post-processing (sheet music only) ───────────────────────────────────────

MIN_NOTE_DURATION = 0.125
QUANTISE_GRID = 0.25


def smooth_tempo(score: stream.Score) -> float:
    marks = score.flatten().getElementsByClass(tempo.MetronomeMark)
    bpms = [m.number for m in marks if m.number and 40 <= m.number <= 240]
    chosen_bpm = float(sorted(bpms)[len(bpms) // 2]) if bpms else 120.0

    for part in score.parts:
        for m in part.flatten().getElementsByClass(tempo.MetronomeMark):
            m.activeSite.remove(m)
    score.parts[0].measure(0 if score.parts[0].measure(0) else 1).insert(
        0, tempo.MetronomeMark(number=chosen_bpm)
    )
    logger.info(f"Tempo smoothed to {chosen_bpm} BPM")
    return chosen_bpm


def detect_and_set_key(score: stream.Score) -> key.Key:
    detected = score.analyze("key")
    logger.info(f"Detected key: {detected}")
    ks = detected.asKey() if hasattr(detected, "asKey") else key.Key(detected.tonic.name, detected.mode)
    for part in score.parts:
        measures = part.getElementsByClass(stream.Measure)
        if measures:
            measures[0].insert(0, ks)
        for n in part.flatten().getElementsByClass(note.Note):
            try:
                n.pitch.simplifyEnharmonic(inPlace=True, mostCommon=True)
            except Exception:
                pass
    return detected


def infer_time_signature(score: stream.Score) -> str:
    existing = score.flatten().getElementsByClass(meter.TimeSignature)
    if existing:
        ts_str = existing[0].ratioString
        logger.info(f"Keeping existing time signature: {ts_str}")
        return ts_str

    ts_str = "4/4"
    try:
        notes_per_measure = [
            len(m.flatten().getElementsByClass(note.Note))
            for part in score.parts
            for m in part.getElementsByClass(stream.Measure)
            if len(m.flatten().getElementsByClass(note.Note)) > 0
        ]
        if notes_per_measure:
            avg = sum(notes_per_measure) / len(notes_per_measure)
            ts_str = "3/4" if avg <= 3 else "6/8" if avg >= 6 else "4/4"
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
    try:
        score = score.quantize(
            quarterLengthDivisors=[1.0 / grid],
            processOffsets=True,
            processDurations=True,
            inPlace=False,
        )
        logger.info(f"Score quantised to {grid} QL grid")
    except Exception as e:
        logger.warning(f"Quantisation failed, skipping: {e}")
    return score


def merge_short_rests(score: stream.Score, min_rest: float = 0.25) -> stream.Score:
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
    logger.info("Starting score post-processing...")
    score = filter_short_notes(score)
    score = quantise_score(score)
    score = merge_short_rests(score)
    try:
        smooth_tempo(score)
    except Exception as e:
        logger.warning(f"Tempo smoothing failed: {e}")
    try:
        detect_and_set_key(score)
    except Exception as e:
        logger.warning(f"Key detection failed: {e}")
    try:
        infer_time_signature(score)
    except Exception as e:
        logger.warning(f"Time signature inference failed: {e}")
    try:
        score = score.makeNotation(inPlace=False)
    except Exception as e:
        logger.warning(f"makeNotation failed: {e}")
    logger.info("Post-processing complete")
    return score


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


# ── Chords: chromagram extraction ─────────────────────────────────────────────

def extract_chords_from_audio(audio_path: str) -> list[dict]:
    """
    Chromagram-based chord recognition using librosa.

    Uses 0.5-second CQT frames so chord changes at normal song tempos are
    captured, then applies a 5-frame median filter to smooth out noise before
    template matching. Consecutive duplicate labels are collapsed so the output
    contains one entry per distinct chord segment.
    """
    logger.info("Running chroma-based chord recognition...")

    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # 0.5-second hop: fine enough for typical chord changes (1–2 beats at
    # 120 BPM ≈ 0.5–1 s) without being so short that noise dominates.
    hop_length = sr // 2

    # CQT chroma has better frequency resolution than STFT chroma and
    # captures low-register instruments (bass, guitar) more reliably.
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # Median-filter across 5 frames (~2.5 s) to smooth transients while
    # keeping genuine chord changes sharper than CENS smoothing would.
    from scipy.ndimage import median_filter
    chroma = median_filter(chroma, size=(1, 5))

    chords_out: list[dict] = []
    prev_label: str | None = None

    for frame_idx in range(chroma.shape[1]):
        frame = chroma[:, frame_idx]
        norm = float(np.linalg.norm(frame))
        if norm < 0.01:  # near-silence
            continue
        frame = frame / norm

        best_label, best_score = None, 0.0
        for label, template in _TEMPLATES.items():
            score = float(np.dot(frame, template))
            if score > best_score:
                best_score = score
                best_label = label

        # Lower threshold catches more valid chords; dedup handles repeats.
        if best_label is None or best_score < 0.65 or best_label == prev_label:
            continue

        timestamp = librosa.frames_to_time(frame_idx, sr=sr, hop_length=hop_length)
        chords_out.append({"chord": best_label, "timestamp": round(float(timestamp), 2)})
        prev_label = best_label

    logger.info(f"Detected {len(chords_out)} chords via chroma analysis")
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
    Downloads audio (cached), runs Basic Pitch (cached), post-processes and
    converts MIDI → MusicXML via music21.
    Returns JSON: { song_name, artist, musicxml }
    """
    work_dir = tempfile.mkdtemp(prefix="song2sheet_")
    try:
        midi_path = get_midi(request.song_name, request.artist, work_dir)
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
    Downloads audio (cached), runs chromagram-based chord recognition,
    fetches synced lyrics from lrclib and aligns chords to lyric lines.
    Returns JSON: { song_name, artist, has_lyrics, lines, all_chords }
    """
    work_dir = tempfile.mkdtemp(prefix="song2chords_")
    try:
        audio_path = get_audio(request.song_name, request.artist, work_dir)
        chords = extract_chords_from_audio(audio_path)
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
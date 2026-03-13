import os
import subprocess
import tempfile
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
import music21

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Song to Sheet Music API",
    description="Converts a song (by name and artist) to MusicXML sheet music using Basic Pitch",
    version="2.0.0",
)


class SongRequest(BaseModel):
    song_name: str
    artist: str


def search_and_download_audio(song_name: str, artist: str, output_dir: str) -> str:
    """
    Search YouTube for the song and download as MP3 using the yt-dlp CLI.
    Returns the path to the downloaded MP3.
    """
    query = f"ytsearch1:{artist} - {song_name}"
    mp3_path = os.path.join(output_dir, "audio.mp3")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "192K",
        "--no-playlist",
        "--match-filter", "duration < 600",
        "--retries", "3",
        "-o", mp3_path,
        query,
    ]

    logger.info(f"Running yt-dlp for: {artist} - {song_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"yt-dlp failed:\n{result.stderr}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download audio from YouTube: {result.stderr.strip()}",
        )

    if not os.path.exists(mp3_path):
        raise HTTPException(
            status_code=500,
            detail="yt-dlp exited successfully but MP3 file was not found.",
        )

    logger.info(f"Audio downloaded to: {mp3_path}")
    return mp3_path


def convert_audio_to_midi(mp3_path: str, output_dir: str) -> str:
    """Run Basic Pitch inference and return the path to the generated MIDI file."""
    predict_and_save(
        audio_path_list=[mp3_path],
        output_directory=output_dir,
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )

    stem = Path(mp3_path).stem
    midi_path = os.path.join(output_dir, f"{stem}_basic_pitch.mid")

    if not os.path.exists(midi_path):
        mid_files = list(Path(output_dir).glob("*.mid"))
        if not mid_files:
            raise HTTPException(
                status_code=500,
                detail="MIDI conversion failed — no MIDI file was produced.",
            )
        midi_path = str(mid_files[0])

    logger.info(f"MIDI written to: {midi_path}")
    return midi_path


def convert_midi_to_musicxml(midi_path: str, output_dir: str) -> str:
    """
    Convert a MIDI file to MusicXML using music21.
    Returns the MusicXML string.
    """
    logger.info(f"Converting MIDI to MusicXML: {midi_path}")

    score = music21.converter.parse(midi_path)

    # Attempt key and time signature inference for cleaner notation
    score = score.makeNotation()

    xml_path = os.path.join(output_dir, "score.musicxml")
    score.write("musicxml", fp=xml_path)

    with open(xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    logger.info("MusicXML conversion complete")
    return xml_content


def _delete_dir(path: str) -> None:
    import shutil
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


@app.post("/convert", summary="Convert a song to sheet music (MusicXML)")
async def convert_song_to_sheet_music(request: SongRequest):
    """
    Accepts a song name and artist, downloads audio from YouTube,
    converts it to MIDI via Basic Pitch, then to MusicXML via music21.
    Returns JSON with the MusicXML string for frontend rendering.
    """
    work_dir = tempfile.mkdtemp(prefix="song2sheet_")

    try:
        mp3_path = search_and_download_audio(request.song_name, request.artist, work_dir)
        midi_path = convert_audio_to_midi(mp3_path, work_dir)

        # Delete MP3 immediately after MIDI is produced
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
            logger.info(f"Deleted MP3: {mp3_path}")

        musicxml = convert_midi_to_musicxml(midi_path, work_dir)

        return JSONResponse(content={
            "song_name": request.song_name,
            "artist": request.artist,
            "musicxml": musicxml,
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during conversion")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        # Always clean up temp directory
        _delete_dir(work_dir)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
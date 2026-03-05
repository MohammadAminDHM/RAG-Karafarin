import json
import os
import re
import wave

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from vosk import KaldiRecognizer, Model

from app.config import settings
from app.utils.audio import preprocess_audio

router = APIRouter()

# ── Load model once at startup ────────────────────────────────────────────────
model_path = os.path.join(settings.VOSK_MODEL_DIR, settings.VOSK_MODEL_NAME)
if not os.path.exists(model_path):
    raise RuntimeError(f"Vosk model not found at {model_path}. Run the download script first.")

model = Model(model_path)

# Larger chunks give the acoustic model more context per step → fewer split words
_CHUNK_FRAMES = 8000

# Single-character tokens to keep even when isolated (common Persian prepositions)
_KEEP_SINGLE = {"و", "در", "از", "با", "به", "که", "را", "تا", "هم", "یا", "اگر"}


def _clean_transcript(text: str) -> str:
    """
    Post-process Vosk output for cleaner Persian text:
      - Collapse multiple whitespace
      - Drop isolated single characters that are almost always noise
        (but keep common Persian function words)
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    cleaned = [t for t in tokens if len(t) > 1 or t in _KEEP_SINGLE]
    return " ".join(cleaned)


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_path: str | None = None
    wf = None
    try:
        raw_bytes = await file.read()
        audio_path = preprocess_audio(raw_bytes, file.filename or "audio.wav")

        wf = wave.open(audio_path, "rb")
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getframerate() != settings.SAMPLE_RATE
        ):
            raise HTTPException(
                status_code=400,
                detail="Audio must be mono 16-bit PCM WAV at 16 000 Hz",
            )

        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        segments: list[str] = []
        while True:
            data = wf.readframes(_CHUNK_FRAMES)
            if not data:
                break
            if rec.AcceptWaveform(data):
                seg = json.loads(rec.Result()).get("text", "").strip()
                if seg:
                    segments.append(seg)

        final = json.loads(rec.FinalResult()).get("text", "").strip()
        if final:
            segments.append(final)

        transcript = _clean_transcript(" ".join(segments))
        return JSONResponse(content={"transcript": transcript})

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if wf:
            wf.close()
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)

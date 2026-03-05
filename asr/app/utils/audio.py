import io
import os
import tempfile

from pydub import AudioSegment
from pydub.effects import normalize

from app.config import settings

ALLOWED_AUDIO_FORMATS = {"wav", "mp3", "m4a", "ogg", "flac", "aac", "webm", "3gp"}

# Short silence padding around the recording helps Vosk detect word boundaries
_SILENCE_PAD_MS = 300


def _pad_silence(audio: AudioSegment) -> AudioSegment:
    silence = AudioSegment.silent(duration=_SILENCE_PAD_MS, frame_rate=audio.frame_rate)
    return silence + audio + silence


def preprocess_audio(input_bytes: bytes, filename: str) -> str:
    """
    Convert any supported audio to mono 16-kHz 16-bit WAV ready for Vosk.
    Steps:
      1. Detect format from extension (fall back to 'wav')
      2. Trim to MAX_AUDIO_DURATION
      3. Convert to mono + 16 kHz
      4. Normalize volume  (quiet mic recordings become clearly audible)
      5. Pad short silence at both ends (improves Vosk word-boundary detection)
      6. Export to a temp WAV and return its path
    """
    ext = os.path.splitext(filename)[-1].lower().lstrip(".")
    if not ext:
        ext = "wav"
    if ext not in ALLOWED_AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: .{ext}")

    audio = AudioSegment.from_file(io.BytesIO(input_bytes), format=ext)

    # Trim
    max_ms = settings.MAX_AUDIO_DURATION * 1000
    if len(audio) > max_ms:
        audio = audio[:max_ms]

    # Mono + target sample rate
    audio = audio.set_channels(1).set_frame_rate(settings.SAMPLE_RATE)

    # Volume normalisation — makes quiet recordings audible to the model
    audio = normalize(audio)

    # Silence padding
    audio = _pad_silence(audio)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio.export(f.name, format="wav")
        return f.name

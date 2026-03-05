from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    VOSK_MODEL_NAME = os.getenv("VOSK_MODEL_NAME", "vosk-model-fa-0.5")
    VOSK_MODEL_DIR = os.getenv("VOSK_MODEL_DIR", "models")
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
    MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", 4500))
    PORT = int(os.getenv("PORT", 8080))
    HOST = os.getenv("HOST", "0.0.0.0")

settings = Settings()

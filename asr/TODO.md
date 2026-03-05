
# 📝 TODO – Persian ASR with Vosk (Offline, FastAPI, Streamlit)

This project builds an offline Persian speech-to-text (ASR) system using the Vosk model, FastAPI for the backend, and Streamlit for the frontend.

---

## ✅ 1) Create `.venv`

* [X] Install [uv](https://github.com/astral-sh/uv) or use built-in `venv`

* [X] Create a Python 3.11 virtual environment:

  ```bash
  uv venv --python=3.11
  source .venv/bin/activate
  ```

* [X] Install dependencies:

  ```bash
  uv pip install -r requirements.txt
  ```

---

## ✅ 2) Create Project Structure

* [X] Initialize directory layout:

  ```
  .
  ├── app/                     # FastAPI backend
  │   ├── main.py              # Entry point
  │   ├── api.py               # API logic
  │   └── utils/               # Audio processing helpers
  ├── ui/                      # Streamlit frontend
  │   └── app.py
  ├── scripts/
  │   └── download_vosk_model.py
  ├── models/                  # Vosk models
  ├── .env                     # Environment variables
  ├── requirements.txt
  └── TODO.md
  └── README.md

  ```

---

## ✅ 3) Define `.env` Configuration

* [X] Create a `.env` file:

  ```env
  VOSK_MODEL_NAME=vosk-model-fa-0.5
  VOSK_MODEL_DIR=models
  SAMPLE_RATE=16000
  MAX_AUDIO_DURATION=60
  PORT=8080
  HOST=0.0.0.0
  ```

* [X] Load env vars in both backend and UI using `dotenv`

---

## ✅ 4) Download & Load Vosk Large Model

* [X] Add script: `scripts/download_vosk_model.py`

  * Downloads model if not already available

* [X] Automatically load model on app startup

---

## ✅ 5) Receive & Preprocess Audio

* [X] Accept `.wav`, `.mp3`, `.m4a`, etc. via HTTP or frontend
* [X] Convert audio to:

  * Mono
  * 16kHz sample rate
* [X] Enforce audio duration limit (`MAX_AUDIO_DURATION`)
* [X] Save temporary files securely if needed

---

## ✅ 6) Build FastAPI ASR Endpoint

* [X] `POST /transcribe`:

  * Accepts `multipart/form-data` with audio file
  * Returns:

    ```json
    {
      "text": "recognized text",
      "words": [
        {"word": "سلام", "start": 0.23, "end": 0.56, "conf": 0.89}
      ],
      "meta": {
        "model": "vosk-model-fa-0.5",
        "duration": 4.2,
        "sample_rate": 16000
      }
    }
    ```

* [X] Add `GET /health` for service check

* [X] Handle common errors:

  * `415`: unsupported file format
  * `413`: file too large
  * `422`: missing audio field

---

## ✅ 7) Build Streamlit UI

* [ ] Upload or record voice input
* [ ] Send audio to `/transcribe` endpoint
* [ ] Display:

  * Transcribed text
  * Word-level timestamps
  * Audio waveform or playback
* [ ] (Optional) Send transcribed text to chatbot backend
* [ ] (Optional) Display chatbot reply with TTS or subtitles

---

## 🔄 Extras (Optional Features)

* [ ] Dockerfile + docker-compose
* [ ] Whisper fallback model
* [ ] WebSocket for real-time transcription
* [ ] Model benchmark comparisons
* [ ] VAD / noise removal
* [ ] Subtitle (SRT/VTT) export
* [ ] Streamlit audio player with timeline

---

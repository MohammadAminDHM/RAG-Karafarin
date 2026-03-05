# 🗣️ Persian ASR with Vosk

This project provides an offline Automatic Speech Recognition (ASR) system for the Persian language using Vosk, FastAPI, and Streamlit.

Run `init_project.sh` to set up the base structure.
Then follow `TODO.md` to build and extend the system.

## Save Docker images (export to .tar)

This project builds two Docker images via `docker-compose.yml`:

- `stt-web:latest`
- `stt-download_model:latest`

On **Windows (PowerShell)** you can export them to `docker-images/` like this:

```powershell
.\scripts\save_docker_images.ps1
```

The exported files will be:

- `docker-images/stt-web.tar`
- `docker-images/stt-download_model.tar`

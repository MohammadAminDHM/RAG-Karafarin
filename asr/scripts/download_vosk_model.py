import os
import time
import zipfile
import urllib.request
import urllib.error
from app.config import settings


model_name = settings.VOSK_MODEL_NAME
model_dir = settings.VOSK_MODEL_DIR
download_url = f"https://alphacephei.com/vosk/models/{model_name}.zip"

CHUNK_SIZE = 1024 * 1024  # 1MB

def _remote_filesize(url: str) -> int | None:
    """Best-effort remote size (bytes). Returns None if unavailable."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None

def _download_with_resume(url: str, dest_path: str, max_retries: int = 8) -> None:
    """
    Download with retries and resume support (Range requests).
    This avoids urllib.urlretrieve() which fails hard on large/unstable downloads.
    """
    expected_size = _remote_filesize(url)
    attempt = 0

    while True:
        existing = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        if expected_size and existing >= expected_size:
            return

        headers = {}
        file_mode = "wb"
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            file_mode = "ab"

        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", None)
                if existing > 0 and status == 200:
                    # Server ignored Range; restart from scratch.
                    existing = 0
                    file_mode = "wb"

                downloaded = existing
                with open(dest_path, file_mode) as f:
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

            final_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
            if expected_size and final_size < expected_size:
                raise urllib.error.ContentTooShortError(
                    f"incomplete download: got {final_size} out of {expected_size} bytes", None
                )
            return

        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            wait_s = min(60, 2 ** attempt)
            print(
                f"⚠️ Download interrupted ({type(e).__name__}: {e}). "
                f"Retrying in {wait_s}s (attempt {attempt}/{max_retries})..."
            )
            time.sleep(wait_s)

def download_and_extract():
    model_path = os.path.join(model_dir, model_name)
    zip_path = f"{model_path}.zip"

    if os.path.exists(model_path):
        print(f"✅ Model already exists at {model_path}")
        return

    print(f"⬇️ Downloading model from: {download_url}")
    os.makedirs(model_dir, exist_ok=True)
    _download_with_resume(download_url, zip_path)

    print("📦 Extracting model...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        bad_file = zip_ref.testzip()
        if bad_file is not None:
            raise RuntimeError(f"Downloaded zip is corrupted (first bad file: {bad_file}). Please retry.")
        zip_ref.extractall(model_dir)

    os.remove(zip_path)
    print(f"✅ Model extracted to: {model_path}")

if __name__ == "__main__":
    download_and_extract()

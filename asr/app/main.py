import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from app.api import router
import uvicorn

app = FastAPI(
    title="🗣 Persian ASR API",
    description="Offline ASR using Vosk for Persian",
    version="0.1.0",
)



# Include transcribe routes
app.include_router(router)

# Optional root route
@app.get("/")
async def root():
    return {"message": "Welcome to Persian ASR API ✨"}

# Optional health check
@app.get("/health")
async def health():
    return {"status": "ok"}

# Optional: run directly with `python app/main.py`
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=433)

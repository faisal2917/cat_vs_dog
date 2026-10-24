"""Web app: upload a photo in the browser and the trained model says cat or dog.

Run:   uvicorn app:app --port 8000      (or: docker compose up)
Open:  http://localhost:8000

API:   POST /predict with an image as form field "file"
       -> {"label": "cat", "confidence": 0.97, "probabilities": {"cat": 0.97, "dog": 0.03}}

Uses the newest model in models/, or the file named by the MODEL_PATH environment variable.
"""
import io
import os
import warnings
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from predict import find_model, load_model
from train import ROOT

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 40_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)  # reject huge images instead of just warning
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' blob:; frame-ancestors 'none'; "
                               "base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

model_path = Path(os.environ["MODEL_PATH"]) if os.environ.get("MODEL_PATH") else find_model()
model, classes, transform = load_model(model_path)

app = FastAPI(title="Cat vs Dog", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.update(SECURITY_HEADERS)
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": model_path.name}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Not an image file, or the image is damaged or too large")
    with torch.inference_mode():
        probs = model(transform(image).unsqueeze(0)).softmax(1)[0]
    idx = int(probs.argmax())
    return {"label": classes[idx], "confidence": round(probs[idx].item(), 4),
            "probabilities": {c: round(p, 4) for c, p in zip(classes, probs.tolist())}}

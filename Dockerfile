FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# CPU-only PyTorch first: the default Linux wheels pull in ~3 GB of CUDA libraries
COPY requirements.txt .
RUN grep -E "^torch(vision)?==" requirements.txt \
        | xargs pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# run as a normal user; the cache folder lets train.py store the pretrained weights
RUN useradd --create-home app && mkdir -p /home/app/.cache/torch && chown -R app /home/app/.cache
COPY *.py ./
COPY static ./static
USER app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]

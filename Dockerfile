# Render's native Python runtime gives no apt access, and Docling's
# `[standard]` extra pulls in torch, onnxruntime, rapidocr and opencv, several
# of which dynamically link system libraries a bare python:slim image doesn't
# ship (libgomp for torch/onnxruntime's OpenMP threading, libGL/glib for
# opencv's image ops used by docling-ibm-models' layout model). That's why
# this is a Dockerfile-based Render deploy rather than the native Python
# runtime - see render.yaml.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Shell form (not exec-array) so $PORT - set by Render at runtime - actually expands.
CMD ["sh", "-c", "uvicorn app.platform.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

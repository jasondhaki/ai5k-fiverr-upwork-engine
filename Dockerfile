# PDF_PARSER defaults to pypdfium2 (see app/ingestion/pdf_extractor.py),
# which needs none of this - only requirements.txt is installed below, and
# that alone is enough to run the app as deployed today. These system
# libraries exist ONLY for the PDF_PARSER=docling production path
# (docling's `[standard]` extra pulls in torch, onnxruntime, rapidocr and
# opencv, several of which dynamically link libgomp for OpenMP threading and
# libGL/glib for opencv's image ops) - they're a few MB, not the problem, so
# they stay installed defensively rather than ripping out the Docker-based
# deploy over a dependency this image doesn't even install by default right
# now. Switching PDF_PARSER=docling into production also needs
# requirements-production.txt installed - NOT done here yet; add a
# `RUN pip install --no-cache-dir -r requirements-production.txt` line when
# actually making that switch (and size the Render instance for it - see
# that file's comment on peak memory).
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

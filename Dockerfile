FROM python:3.11-slim

WORKDIR /app

# ChromaDB (sqlite3) + pymupdf4llm + sentence-transformers derleme gereksinimleri
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libglib2.0-0 \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch (~200MB) — CUDA olmayan ortamlar için
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Not: PDF ingestion (RAG/ingest.py) container dışında veya ayrı bir job olarak çalıştırılmalı.
# PDF_DIR ortam değişkeniyle klasör yolu override edilebilir.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

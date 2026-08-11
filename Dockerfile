FROM python:3.11-slim

WORKDIR /app

# pdfplumber ve sentence-transformers için derleme araçları
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Önce CPU-only torch (CUDA olmadan ~200MB, CUDA ile ~2.5GB fark var)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

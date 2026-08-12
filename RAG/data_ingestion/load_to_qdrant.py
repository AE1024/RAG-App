# PDF → Chunker → Embedding → Qdrant

import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from data_ingestion.chunker import Chunker
from embedding import EmbeddingGenerator

QDRANT_URL = "http://localhost:6333"
COLLECTION = "mevzuat"
PDF_DIR = Path(r"C:\Users\elmaz\Desktop\mevzautlar_kanunlar")

if not PDF_DIR.exists():
    print(f"Hata: Klasör bulunamadı → {PDF_DIR}")
    sys.exit(1)

client = QdrantClient(url=QDRANT_URL)
chunker = Chunker()
embedding_generator = EmbeddingGenerator()

# Koleksiyonu oluştur (varsa sıfırla)
client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=embedding_generator.vector_dim, distance=Distance.COSINE),
)
print(f"Koleksiyon hazır: '{COLLECTION}' (boyut={embedding_generator.vector_dim})")

point_id = 0

for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
    print(f"\nİşleniyor: {pdf_file.name}")

    text = chunker.extract_text_from_pdf(str(pdf_file))
    chunked = chunker.split_by_article(text, law_name=pdf_file.stem)

    if not chunked:
        print("  Chunk bulunamadı, atlanıyor.")
        continue

    embeddings = embedding_generator.build_index(chunked, batch_size=64)

    points = [
        PointStruct(
            id=point_id + i,
            vector=embeddings[i].tolist(),
            payload={
                "kanun_adi": chunk["kanun_adi"],
                "madde": chunk.get("madde", "Giriş"),
                "metin": chunk["metin"],
                "kaynak": pdf_file.name,
            },
        )
        for i, chunk in enumerate(chunked)
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    point_id += len(points)
    print(f"{len(points)} chunk Qdrant'a yüklendi.")

print(f"\nToplam {point_id} chunk kaydedildi → koleksiyon: '{COLLECTION}'")

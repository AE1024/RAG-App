import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from data_ingestion.chunker import Chunker


class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        print(f"Model yükleniyor: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # Boyutu güvenli bir şekilde test edip alıyoruz
        sample_vector = self.model.encode("test", convert_to_numpy=True)
        self.vector_dim = sample_vector.shape[0]
        print(f"Model yüklendi. Vektör Boyutu: {self.vector_dim}")

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Bir metin grubunu (batch) vektörleştirir."""
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def build_index(self, chunked_texts: list[dict], batch_size: int = 256) -> np.ndarray:
        """Tüm corpus'u batch'ler halinde ilerleme çubuğuyla (tqdm) vektörleştirir."""
        if not chunked_texts:
            return np.array([], dtype=np.float32).reshape(0, self.vector_dim)

        texts = [chunk['metin'] for chunk in chunked_texts]
        chunks = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (Vektörleştiriliyor)"):
            batch_texts = texts[i : i + batch_size]
            chunks.append(self._embed_batch(batch_texts))
            
        return np.vstack(chunks)

    def save_and_normalize(self, embeddings: np.ndarray, save_path: str):
        """Vektör matrisini diske kaydeder ve kosinüs benzerliği için normalize eder."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Ham matrisi kaydet
        np.save(path, embeddings)
        print(f"Embedding matrisi başarıyla kaydedildi: {path}")

        # 2. Ön-normalizasyon yap
        embeddings_normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings_normed

chunker = Chunker()

text = chunker.extract_text_from_pdf("C:\\Users\\elmaz\\Desktop\\mevzautlar_kanunlar\\guvenliksorusturma.pdf")
chunked = chunker.split_by_article(text, law_name="Güvenlik Soruşturması ve Arşiv Araştırması Kanunu")

embedding_generator = EmbeddingGenerator()
embeddings = embedding_generator.build_index(chunked, batch_size=256)
normalized_embeddings = embedding_generator.save_and_normalize(embeddings, "C:\\Users\\elmaz\\Desktop\\mevzautlar_kanunlar\\guvenliksorusturma_embeddings.npy")

for i, embd in enumerate(normalized_embeddings):
    print(f"Madde {i+1} -> Vektör Boyutu: {embd.shape[0]}")
    
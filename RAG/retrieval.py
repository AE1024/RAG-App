"""ChromaDB parent-child retriever.

Akış:
  sorgu
    → child chunk'lar arasında similarity ara  (küçük, hassas parçalar)
    → score < 0.45 olanları at               (A iyileştirmesi)
    → child'ların parent_id'lerini çöz        (B iyileştirmesi)
    → parents.json'dan tam madde metinlerini getir
    → LLM'e tam madde içerikleri gider (bölünmüş parça değil)
"""

import json
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from RAG.embedding import E5SmallEmbeddings
from backend.tracing import traceable

CHROMA_DIR   = str(Path(__file__).resolve().parent.parent / "chroma_db")
PARENTS_FILE = Path(__file__).resolve().parent.parent / "chroma_db" / "parents.json"

# A iyileştirmesi: 0.45 → 0.52  (kenar maddeleri eler, sadece gerçekten ilgili olanlar geçer)
_SCORE_THRESHOLD = 0.52

_NORMALIZATIONS = [
    (re.compile(r'\bzaman\s+aş[ıi]m[ıi]\b',           re.IGNORECASE), "zamanaşımı"),
    (re.compile(r'\bhak\s+kullan[ıi]m[ıi]\b',          re.IGNORECASE), "hak kullanımı"),
    (re.compile(r'\bborç\s+ili[şs]kisi\b',             re.IGNORECASE), "borç ilişkisi"),
    (re.compile(r'\bsözle[şs]me\s+özgürlü[ğg]ü\b',    re.IGNORECASE), "sözleşme özgürlüğü"),
    (re.compile(r'\bhaks[ıi]z\s+fiil\b',               re.IGNORECASE), "haksız fiil"),
    (re.compile(r'\bsebepsiz\s+zenginle[şs]me\b',      re.IGNORECASE), "sebepsiz zenginleşme"),
]


def _normalize_query(query: str) -> str:
    for pattern, replacement in _NORMALIZATIONS:
        query = pattern.sub(replacement, query)
    return query


class Retriever:
    def __init__(self):
        print("ChromaDB (parent-child) yükleniyor...")

        embedder = E5SmallEmbeddings()
        self.db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedder,
            collection_name="mevzuat",
        )

        self._parents: dict = {}
        if PARENTS_FILE.exists():
            self._parents = json.loads(PARENTS_FILE.read_text(encoding="utf-8"))
            print(f"Hazır. ({len(self._parents)} parent yüklendi)")
        else:
            print(
                "UYARI: parents.json bulunamadı. "
                "Lütfen çalıştırın: python -m RAG.data_ingestion.ingest --reset"
            )

    @traceable(name="retrieval", run_type="retriever")
    def search(self, query: str, top_k: int = 4) -> list[Document]:
        normalized = _normalize_query(query)

        # 1. Küçük child chunk'lar arasında similarity search (sadece child'lar)
        docs_with_scores = self.db.similarity_search_with_relevance_scores(
            normalized,
            k=60,
            filter={"doc_type": "child"},
        )

        # 2. Score threshold filtresi
        passing = [
            (doc, score)
            for doc, score in docs_with_scores
            if score >= _SCORE_THRESHOLD
        ]

        # Hiçbiri eşiği geçmezse en iyi top_k ile devam et
        if not passing:
            passing = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)[:top_k]

        # 3. Her parent'ın en yüksek child skorunu tut (parent dedup)
        best_score: dict[str, float] = {}
        for doc, score in passing:
            pid = doc.metadata.get("parent_id", "")
            if pid and score > best_score.get(pid, -1.0):
                best_score[pid] = score

        # Skora göre sıralı parent ID listesi
        ranked_pids = sorted(best_score, key=lambda pid: best_score[pid], reverse=True)[:top_k]

        # 4. parents.json'dan tam madde metnini getir
        results: list[Document] = []
        for pid in ranked_pids:
            entry = self._parents.get(pid)
            if entry:
                results.append(Document(
                    page_content=entry["content"],
                    metadata=entry["metadata"],
                ))

        return results

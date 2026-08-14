"""LangChain + ChromaDB retriever — MMR."""

import os
import re
from pathlib import Path

from langchain_chroma import Chroma

from RAG.embedding import E5SmallEmbeddings

CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")

_NORMALIZATIONS = [
    (re.compile(r'\bzaman\s+aş[ıi]m[ıi]\b',           re.IGNORECASE), 'zamanaşımı'),
    (re.compile(r'\bhak\s+kullan[ıi]m[ıi]\b',          re.IGNORECASE), 'hak kullanımı'),
    (re.compile(r'\bborç\s+ili[şs]kisi\b',             re.IGNORECASE), 'borç ilişkisi'),
    (re.compile(r'\bsözle[şs]me\s+özgürlü[ğg]ü\b',    re.IGNORECASE), 'sözleşme özgürlüğü'),
    (re.compile(r'\bhaks[ıi]z\s+fiil\b',               re.IGNORECASE), 'haksız fiil'),
    (re.compile(r'\bsebepsiz\s+zenginle[şs]me\b',      re.IGNORECASE), 'sebepsiz zenginleşme'),
]


def _normalize_query(query: str) -> str:
    for pattern, replacement in _NORMALIZATIONS:
        query = pattern.sub(replacement, query)
    return query


class Retriever:
    def __init__(self):
        print("ChromaDB + MMR yükleniyor...")

        embedder = E5SmallEmbeddings()
        self.db  = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedder,
            collection_name="mevzuat",
        )

        # MMR: fetch_k kadar aday çek, çeşitlilik filtresiyle k'ya indir
        self.retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.7},
        )

        print("Hazır.")

    def search(self, query: str, top_k: int = 5) -> list:
        normalized = _normalize_query(query)
        docs = self.retriever.invoke(normalized)
        return docs[:top_k]

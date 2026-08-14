"""PDF → Markdown → ChromaDB ingestion pipeline (LangChain)."""

import re
import sys
from pathlib import Path

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from RAG.embedding import E5SmallEmbeddings

PDF_DIR    = Path(r"C:\Users\elmaz\Desktop\mevzautlar_kanunlar")
CHROMA_DIR = str(Path(__file__).resolve().parent.parent.parent / "chroma_db")

# pymupdf4llm'in ürettiği Markdown başlık seviyeleri
HEADERS = [
    ("#",   "bolum"),
    ("##",  "madde"),
    ("###", "fikra"),
]

# Regex: "MADDE 22", "Madde 22 -", "Md. 22" gibi kalıpları yakala
_MADDE_RE = re.compile(
    r'(?:MADDE|Madde|madde|MD\.|Md\.)\s*(\d+)',
    re.IGNORECASE,
)

# Dosya adı → okunabilir kanun adı (LLM doğru atıf yapabilsin)
KANUN_ADLARI: dict[str, str] = {
    "iskanunu":                     "İş Kanunu (4857)",
    "turkborclarkanunu":            "Türk Borçlar Kanunu (6098)",
    "turkcezakanunu":               "Türk Ceza Kanunu (5237)",
    "turktüketicikoruma":           "Tüketicinin Korunması Hakkında Kanun (6502)",
    "turkticaretkodeksi":           "Türk Ticaret Kanunu (6102)",
    "medenikanun":                  "Türk Medeni Kanunu (4721)",
    "sosyalsigorta_sagliksigorta":  "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu (5510)",
    "isguvenligicanunu":            "İş Sağlığı ve Güvenliği Kanunu (6331)",
    "kiralamakanunu":               "Kira Sözleşmeleri (Türk Borçlar Kanunu)",
    "verasetintikal":               "Veraset ve İntikal Vergisi Kanunu (7338)",
}


def _extract_madde(text: str) -> str:
    """Chunk metninden madde numarasını regex ile çeker."""
    m = _MADDE_RE.search(text)
    return f"Madde {m.group(1)}" if m else "Giriş"


def main(reset: bool = False):
    if not PDF_DIR.exists():
        print(f"Hata: Klasör bulunamadı → {PDF_DIR}")
        sys.exit(1)

    embedder = E5SmallEmbeddings()

    if reset:
        # Koleksiyonu sıfırla — eski verileri temizle
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection("mevzuat")
            print("Koleksiyon sıfırlandı.")
        except Exception:
            pass

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedder,
        collection_name="mevzuat",
    )

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS,
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )

    total = 0
    for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
        print(f"\nİşleniyor: {pdf_file.name}")

        md_text = pymupdf4llm.to_markdown(str(pdf_file))
        if not md_text.strip():
            print("  Metin çıkarılamadı (taranmış PDF?), atlanıyor.")
            continue

        # 1. Markdown başlık yapısına göre böl
        header_docs = header_splitter.split_text(md_text)

        # 2. Uzun parçaları örtüşmeli böl
        docs = text_splitter.split_documents(header_docs)

        # 3. Metadata zenginleştir
        stem        = pdf_file.stem
        kanun_tam   = KANUN_ADLARI.get(stem, stem)
        for doc in docs:
            # Header splitter madde bulamazsa chunk metninden regex ile çek
            if not doc.metadata.get("madde"):
                doc.metadata["madde"] = _extract_madde(doc.page_content)

            doc.metadata["kanun_adi"]  = stem          # dosya adı (query için)
            doc.metadata["kanun_tam"]  = kanun_tam     # okunabilir ad (LLM için)
            doc.metadata["kaynak"]     = pdf_file.name

        vectorstore.add_documents(docs)
        total += len(docs)
        print(f"  {len(docs)} chunk eklendi.")

    print(f"\nToplam {total} chunk ChromaDB'ye kaydedildi → {CHROMA_DIR}")


if __name__ == "__main__":
    # --reset bayrağı ile eski koleksiyonu sil
    reset = "--reset" in sys.argv
    main(reset=reset)

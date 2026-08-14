"""Markdown → ChromaDB ingestion pipeline — Parent-Child stratejisi.

Akış:
  markdowns/*.md
    → _clean_markdown()        : başlık bloğu + değişiklik notları temizle
    → _split_into_parents()    : her Madde = 1 parent Document
    → _make_children()         : parent'ı ~400 kar'lık child chunk'lara böl
    → ChromaDB                 : child chunk'ları embed edip kaydet
    → parents.json             : parent tam metinlerini ID ile sakla
"""

import json
import re
import sys
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from RAG.embedding import E5SmallEmbeddings

MD_DIR       = Path(__file__).resolve().parent / "markdowns"
CHROMA_DIR   = str(Path(__file__).resolve().parent.parent.parent / "chroma_db")
PARENTS_FILE = Path(CHROMA_DIR) / "parents.json"

# Her "Madde X" ifadesinde madde sınırı (lookahead — metin silinmez)
_MADDE_SPLIT_RE = re.compile(
    r'(?=(?:\*{1,2})?(?:MADDE|Madde)\s+\d+)',
    re.IGNORECASE,
)
_MADDE_NUM_RE = re.compile(r'(?:MADDE|Madde)\s*(\d+)', re.IGNORECASE)

# PDF başlık bloğu: "Kanun Numarası : X  Kabul Tarihi : ..."
_HEADER_BLOCK_RE = re.compile(
    r'\*{0,2}Kanun Numaras[ıi]\s*:.*?(?=\n#|\Z)',
    re.DOTALL | re.IGNORECASE,
)
# Tamamen mülga edilmiş madde başlığı: "Madde 10 – (Mülga: 22/7/1998-4369/82 md.)"
# _DEGISIK_RE çalışmadan önce stub içerikle koru — LLM "kaldırılmıştır" diyebilsin
_MULGA_MADDE_RE = re.compile(
    r'(\*{0,2}(?:MADDE|Madde)\s*\d+[^(\n]*?)\(Mülga\s*:([^)]+)\)',
    re.IGNORECASE,
)
# Satır içi değişiklik/ekleme notları: "(Değişik: ...)", "(Ek: ...)", "(İptal: ...)"
# Dikkat: Mülga burada artık eşleşmeyecek çünkü _MULGA_MADDE_RE önce çalışıyor
_DEGISIK_RE = re.compile(
    r'\((?:Değişik|Ek|İptal)\s*:.*?\)',
    re.IGNORECASE,
)
# Kanun sonundaki değişiklik cetveli tablosu — hukuki içerik taşımaz
# "X SAYILI KANUNA EK VE DEĞİŞİKLİK GETİREN MEVZUATIN..." başlığından dosya sonuna kadar sil
_CETVEL_RE = re.compile(
    r'\*{0,2}\d+\s+SAYILI\s+KANUNA\s+EK\s+VE\s+DEĞİŞİKLİK\s+GETİREN.*',
    re.DOTALL | re.IGNORECASE,
)
# PDF tablo hücrelerindeki HTML <br> artıkları
_BR_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)

KANUN_ADLARI: dict[str, str] = {
    "askerealma":                   "Askerlik Kanunu (1111)",
    "gelirvergisi":                 "Gelir Vergisi Kanunu (193)",
    "guvenliksorusturma":           "Güvenlik Soruşturması ve Arşiv Araştırması Kanunu (7315)",
    "iskanunu":                     "İş Kanunu (4857)",
    "nufushizmet":                  "Nüfus Hizmetleri Kanunu (5490)",
    "sosyalsigorta_sagliksigorta":  "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu (5510)",
    "tuketicihaklari":              "Tüketicinin Korunması Hakkında Kanun (6502)",
    "turkborclarkanunu":            "Türk Borçlar Kanunu (6098)",
}

# Child splitter: küçük ve hassas parçalar — embedding kalitesi için
_child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "],
)


def _clean_markdown(md_text: str) -> str:
    md_text = _HEADER_BLOCK_RE.sub("", md_text)
    md_text = _CETVEL_RE.sub("", md_text)
    # Mülga madde başlıklarını stub içerikle koru (DEGISIK_RE'den önce çalışmalı)
    md_text = _MULGA_MADDE_RE.sub(
        r'\1\n[Bu madde mülga edilmiştir:\2]', md_text
    )
    md_text = _DEGISIK_RE.sub("", md_text)
    md_text = _BR_RE.sub(" ", md_text)
    md_text = re.sub(r"\n{3,}", "\n\n", md_text)
    return md_text.strip()


def _extract_kanun_no(kanun_tam: str) -> str:
    m = re.search(r"\((\d+)\)", kanun_tam)
    return m.group(1) if m else ""


def _split_into_parents(md_text: str, base_metadata: dict) -> list[Document]:
    """Metni madde sınırlarında böler; her madde bir parent Document."""
    parts = _MADDE_SPLIT_RE.split(md_text)
    parents = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 20:
            continue
        m = _MADDE_NUM_RE.search(part)
        madde_label = f"Madde {m.group(1)}" if m else "Giriş"
        meta = {**base_metadata, "madde": madde_label, "doc_type": "parent"}
        parents.append(Document(page_content=part, metadata=meta))
    return parents


def _make_children(parent: Document, parent_id: str) -> list[Document]:
    """Parent maddeyi ~400 kar'lık child chunk'lara böler.

    B iyileştirmesi : child'lar parent_id taşır, retrieval'de tam maddeye ulaşılır.
    A iyileştirmesi : devam chunk'larına madde başlığı eklenir; chunk_index ve
                      is_continuation metadata'ya kaydedilir.
    """
    raw_chunks = _child_splitter.split_text(parent.page_content)
    total = len(raw_chunks)
    children = []
    madde = parent.metadata.get("madde", "")

    for i, chunk_text in enumerate(raw_chunks):
        is_continuation = i > 0
        if is_continuation:
            # LLM ve embedding modeli hangi maddenin devamı olduğunu anlasın
            chunk_text = f"[{madde} — devam {i + 1}/{total}]\n{chunk_text}"

        meta = {
            **parent.metadata,
            "doc_type":        "child",
            "parent_id":       parent_id,
            "chunk_index":     i,
            "is_continuation": is_continuation,
        }
        children.append(Document(page_content=chunk_text, metadata=meta))

    return children


def main(reset: bool = False) -> None:
    if not MD_DIR.exists() or not any(MD_DIR.glob("*.md")):
        print(f"Hata: Markdown dosyası bulunamadı → {MD_DIR}")
        print("Önce şunu çalıştırın: python -m RAG.data_ingestion.pdf_to_markdown")
        sys.exit(1)

    embedder = E5SmallEmbeddings()
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    if reset:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection("mevzuat")
            print("ChromaDB koleksiyonu sıfırlandı.")
        except Exception:
            pass
        if PARENTS_FILE.exists():
            PARENTS_FILE.unlink()
            print("parents.json sıfırlandı.")

    # Mevcut parent store'u yükle (--reset yoksa ekleme modu)
    parents_store: dict = {}
    if PARENTS_FILE.exists():
        parents_store = json.loads(PARENTS_FILE.read_text(encoding="utf-8"))

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedder,
        collection_name="mevzuat",
    )

    total_parents = 0
    total_children = 0

    for md_file in sorted(MD_DIR.glob("*.md")):
        stem      = md_file.stem
        kanun_tam = KANUN_ADLARI.get(stem, stem)
        kanun_no  = _extract_kanun_no(kanun_tam)

        print(f"\nİşleniyor: {md_file.name}")

        md_text = md_file.read_text(encoding="utf-8")
        md_text = _clean_markdown(md_text)

        base_meta = {
            "kanun_adi": stem,
            "kanun_tam": kanun_tam,
            "kanun_no":  kanun_no,
            "kaynak":    stem + ".pdf",
        }

        parents  = _split_into_parents(md_text, base_meta)
        all_children: list[Document] = []

        for parent in parents:
            parent_id = str(uuid.uuid4())
            parents_store[parent_id] = {
                "content":  parent.page_content,
                "metadata": parent.metadata,
            }
            all_children.extend(_make_children(parent, parent_id))

        vectorstore.add_documents(all_children)
        total_parents  += len(parents)
        total_children += len(all_children)
        print(f"  {len(parents)} parent | {len(all_children)} child chunk")

    # Parent store'u kaydet
    PARENTS_FILE.write_text(
        json.dumps(parents_store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nToplam: {total_parents} parent, {total_children} child chunk")
    print(f"ChromaDB   → {CHROMA_DIR}")
    print(f"parents.json → {PARENTS_FILE}")


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    main(reset=reset)

"""PDF dosyalarını ham Markdown dosyalarına çevirir ve markdowns/ klasörüne kaydeder."""

import sys
from pathlib import Path

import pymupdf4llm

PDF_DIR = Path(r"C:\Users\elmaz\Desktop\mevzautlar_kanunlar")
MD_DIR  = Path(__file__).resolve().parent / "markdowns"


def main():
    if not PDF_DIR.exists():
        print(f"Hata: PDF klasörü bulunamadı → {PDF_DIR}")
        sys.exit(1)

    MD_DIR.mkdir(exist_ok=True)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print("Hiç PDF bulunamadı.")
        sys.exit(1)

    converted = 0
    for pdf_file in pdfs:
        out_file = MD_DIR / (pdf_file.stem + ".md")
        if out_file.exists():
            print(f"  Atlanıyor (zaten mevcut): {out_file.name}")
            continue

        print(f"  Dönüştürülüyor: {pdf_file.name} ...", end=" ", flush=True)
        md_text = pymupdf4llm.to_markdown(str(pdf_file))
        out_file.write_text(md_text, encoding="utf-8")
        print("OK")
        converted += 1

    print(f"\n{converted} PDF dönüştürüldü → {MD_DIR}")
    print("İpucu: Markdown kalitesini kontrol ettikten sonra ingest.py çalıştırın.")


if __name__ == "__main__":
    main()

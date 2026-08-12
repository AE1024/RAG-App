import pymupdf 
import pdfplumber
import re
import os
class Chunker:
    def __init__(self ,chunk_size = 1000, chunk_overlap = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.madde_pattern = re.compile(r'(MADDE\s+\d+[\s\–\-]*)')

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        PDF dosyasından metin çıkarır.
        """
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    def split_by_article(self, text: str, law_name: str = "Kanun") -> list[dict]:
        """
        Kanun metnini maddelerine göre parçalar ve her parçaya metadata ekler.
        """
        # Maddelere göre metni bölüyoruz
        splits = self.madde_pattern.split(text)
        chunks = []
        
        # İlk parça genellikle kanunun başlığı/girişidir
        if splits[0].strip():
            chunks.append({
                "kanun_adi": law_name,
                "bolum": "Giriş / Amaç",
                "metin": splits[0].strip()
            })
    # Madde başlıkları ve içeriklerini eşleştirme döngüsü
        for i in range(1, len(splits), 2):
            madde_baslik = splits[i].strip()
            madde_icerik = splits[i+1].strip() if i+1 < len(splits) else ""
            
            full_text = f"{madde_baslik} {madde_icerik}"
            
            # Madde numarasını ayıklama (Örn: "MADDE 4" -> 4)
            madde_no_match = re.search(r'\d+', madde_baslik)
            madde_no = madde_no_match.group(0) if madde_no_match else "Bilinmiyor"

            chunks.append({
                "kanun_adi": law_name,
                "madde": f"Madde {madde_no}",
                "metin": full_text
            })

        return chunks
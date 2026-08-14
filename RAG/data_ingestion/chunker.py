import pymupdf as fitz  # PyMuPDF — pdfplumber'a göre hızlı, çok sütun düzenini doğru okur
import re


class Chunker:
    def __init__(self, min_len: int = 100, max_len: int = 1500, overlap: int = 150):
        self.min_len = min_len
        self.max_len = max_len
        self.overlap = overlap

        # Satır başında "MADDE <n>" + isteğe bağlı ayraç (–, -, —, .)
        self.madde_pattern = re.compile(
            r'(^MADDE\s+\d+\s*[–\-—\.]*\s*)',
            re.IGNORECASE | re.MULTILINE
        )

        # Yeni satırda (1)–(99) + boşluk: gerçek fıkra işaretleri
        # Cümle ortasındaki atıf numaralarını (ör. "Madde 5'in (1) bendi") yakalamaz
        self.fikra_pattern = re.compile(r'\n\s*(?=\(\d{1,2}\)\s)', re.MULTILINE)

    # ------------------------------------------------------------------ #
    #  PDF → ham metin                                                     #
    # ------------------------------------------------------------------ #
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text += page_text + "\n"
        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        # Tek başına satırdaki sayıları (sayfa numaraları) kaldır
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        # Satır sonu boşlukları
        text = re.sub(r'[ \t]+\n', '\n', text)
        # 3+ art arda boş satırı 2'ye indir
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    # ------------------------------------------------------------------ #
    #  Ana bölme: MADDE bazlı                                              #
    # ------------------------------------------------------------------ #
    def split_by_article(self, text: str, law_name: str = "Kanun") -> list[dict]:
        if not text.strip():
            print("  Uyarı: PDF'den metin çıkarılamadı — taranmış görüntü olabilir.")
            return []

        splits = self.madde_pattern.split(text)
        chunks: list[dict] = []

        # splits[0] = MADDE öncesi giriş metni
        giris = splits[0].strip()
        if len(giris) >= self.min_len:
            chunks.append({"kanun_adi": law_name, "madde": "Giriş", "metin": giris})

        # splits[1]="MADDE 5 –", splits[2]="içerik", splits[3]="MADDE 6 –", ...
        for i in range(1, len(splits), 2):
            madde_baslik = splits[i].strip()
            madde_icerik = splits[i + 1].strip() if i + 1 < len(splits) else ""
            full_text    = f"{madde_baslik}\n{madde_icerik}"

            m = re.search(r'\d+', madde_baslik)
            madde_no = m.group(0) if m else "?"

            if len(full_text) < self.min_len:
                continue

            if len(full_text) > self.max_len:
                chunks.extend(self._split_by_fikra(full_text, law_name, madde_no))
            else:
                chunks.append({
                    "kanun_adi": law_name,
                    "madde":     f"Madde {madde_no}",
                    "metin":     full_text,
                })

        return chunks

    # ------------------------------------------------------------------ #
    #  Fıkra bazlı bölme (overlap + madde başlığı korunur)               #
    # ------------------------------------------------------------------ #
    def _split_by_fikra(self, text: str, law_name: str, madde_no: str) -> list[dict]:
        parts = [p.strip() for p in self.fikra_pattern.split(text) if p and p.strip()]

        if len(parts) <= 1:
            return self._split_fixed(text, law_name, madde_no)

        result: list[dict] = []
        # parts[0]: madde başlığı + (varsa) fıkradan önceki giriş cümlesi
        header_ctx = parts[0]

        for part in parts[1:]:
            if result:
                # Önceki chunk'ın sonundan bağlam ekle
                ctx = result[-1]["metin"][-self.overlap:]
            else:
                # İlk fıkraya madde başlığı + giriş metnini ekle
                ctx = header_ctx

            chunk_text = (ctx + "\n" + part).strip()

            if len(chunk_text) < self.min_len:
                # Kısa fıkrayı bir öncekine ekle; yoksa başlık bağlamını büyüt
                if result:
                    result[-1]["metin"] = (result[-1]["metin"] + " " + part).strip()
                else:
                    header_ctx = chunk_text
                continue

            result.append({
                "kanun_adi": law_name,
                "madde":     f"Madde {madde_no}",
                "metin":     chunk_text,
            })

        return result if result else self._split_fixed(text, law_name, madde_no)

    # ------------------------------------------------------------------ #
    #  Sabit boyutlu bölme (fikra yapısı yoksa fallback)                  #
    # ------------------------------------------------------------------ #
    def _split_fixed(self, text: str, law_name: str, madde_no: str) -> list[dict]:
        step = max(1, self.max_len - self.overlap)  # overlap >= max_len olursa kilitlenmeyi önle
        chunks: list[dict] = []
        start = 0
        while start < len(text):
            end        = start + self.max_len
            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.min_len:
                chunks.append({
                    "kanun_adi": law_name,
                    "madde":     f"Madde {madde_no}",
                    "metin":     chunk_text,
                })
            start += step

        return chunks if chunks else [{"kanun_adi": law_name, "madde": f"Madde {madde_no}", "metin": text}]

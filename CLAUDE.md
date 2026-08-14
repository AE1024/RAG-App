# Türk Hukuku RAG Sistemi — Proje Dokümantasyonu

## Projeye Genel Bakış

Bu proje, Türk kanun metinlerini (PDF) işleyip yerel bir LLM (LM Studio) aracılığıyla hukuki sorulara yanıt veren bir **RAG (Retrieval-Augmented Generation)** sistemidir. Tamamen yerel çalışır: internet bağlantısı gerektirmez, veriler dışarı çıkmaz.

---

## Mimari ve Veri Akışı

```
PDF'ler
  │
  ▼
pymupdf4llm → Markdown metni
  │
  ▼
_clean_markdown()         ← başlık bloğu + değişiklik notları temizlenir
  │
  ▼
_split_by_madde()         ← her "Madde X" ifadesinde zorunlu kesim
  │
  ▼
RecursiveCharacterTextSplitter  ← 1400 kar üstü maddeler alt-parçalara bölünür
  │
  ▼
E5SmallEmbeddings         ← "passage: " prefix ile embed
  │
  ▼
ChromaDB (chroma_db/)     ← yerel vektör veritabanı
  │
  ▼  (sorgu zamanı)
E5SmallEmbeddings         ← "query: " prefix ile sorgu embed
  │
  ▼
MMR Retriever (k=8, fetch_k=25)
  + similarity_score_threshold (0.30 altı atılır)
  │
  ▼
build_context()           ← chunk'lar [KanunAdı — MaddeX]\nMetin formatında birleşir
  │
  ▼
LM Studio / Qwen2.5-7B    ← sistem prompt + bağlam + soru
  │
  ▼
FastAPI /chat endpoint    ← JSON response + sources
```

---

## Klasör Yapısı

```
llm_proje/
├── backend/
│   └── main.py              # FastAPI uygulaması, /chat endpoint
├── RAG/
│   ├── embedding.py         # E5SmallEmbeddings sınıfı
│   ├── retrieval.py         # ChromaDB + MMR retriever
│   └── data_ingestion/
│       ├── ingest.py        # PDF → ChromaDB pipeline (ÇALIŞTIR)
│       └── chunker.py       # Eski chunker (artık kullanılmıyor)
├── chroma_db/               # ChromaDB verileri (otomatik oluşur)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                     # LM Studio URL, model adı
```

---

## Dosyalar — Detaylı Açıklama

### `RAG/data_ingestion/ingest.py` — PDF İşleme ve Yükleme

**Çalıştırma:**
```bash
# İlk kez veya sıfırdan yüklemek için:
python -m RAG.data_ingestion.ingest --reset

# Sadece yeni PDF eklemek için (mevcut veriyi silmez):
python -m RAG.data_ingestion.ingest
```

**Ne yapar:**
1. `C:\Users\elmaz\Desktop\mevzautlar_kanunlar\` klasöründeki tüm PDF'leri tarar
2. `pymupdf4llm` ile PDF → Markdown çevirir
3. `_clean_markdown()` ile gürültüyü temizler:
   - PDF başlık bloğunu siler: `"Kanun Numarası : X Kabul Tarihi : ..."`
   - Değişiklik notlarını siler: `"(Değişik: 7/1/2003-4783/1 md.)"`
4. `_split_by_madde()` ile her `Madde X` ifadesinde keser → her madde ayrı chunk
5. 1400 karakterden uzun maddeler `RecursiveCharacterTextSplitter` ile bölünür ama madde numarası korunur
6. Her chunk'a metadata ekler: `kanun_adi`, `kanun_tam`, `madde`, `kaynak`
7. `E5SmallEmbeddings` ile embed edip ChromaDB'ye kaydeder

**`KANUN_ADLARI` mapping (dosya adı → gerçek kanun adı):**
```python
{
    "askerealma":                   "Askerlik Kanunu (1111)",
    "gelirvergisi":                 "Gelir Vergisi Kanunu (193)",
    "guvenliksorusturma":           "Güvenlik Soruşturması ve Arşiv Araştırması Kanunu (7315)",
    "iskanunu":                     "İş Kanunu (4857)",
    "nufushizmet":                  "Nüfus Hizmetleri Kanunu (5490)",
    "sosyalsigorta_sagliksigorta":  "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu (5510)",
    "tuketicihaklari":              "Tüketicinin Korunması Hakkında Kanun (6502)",
    "turkborclarkanunu":            "Türk Borçlar Kanunu (6098)",
}
```
Yeni PDF eklenince buraya da eklenmeli, yoksa LLM dosya adını görüp halüsinasyon yapıyor.

**Neden `_split_by_madde()`?**
`pymupdf4llm` madde başlıklarını `## Madde 4` (heading) değil `**Madde 4**` (bold) olarak üretiyor. Bu yüzden LangChain'in `MarkdownHeaderTextSplitter`'ı madde sınırlarını görmüyor. Regex-based splitter zorunlu oldu.

---

### `RAG/embedding.py` — Embedding Modeli

```python
class E5SmallEmbeddings(HuggingFaceEmbeddings):
    # Model: intfloat/multilingual-e5-small (~120MB, 384 dim)
    # E5 modelleri prefix gerektirir:
    #   - Belge embed: "passage: " + metin
    #   - Sorgu embed: "query: " + sorgu
```

**Model:** `intfloat/multilingual-e5-small`
- İlk çalıştırmada `~/.cache/huggingface/` dizinine indirilir
- 384 boyutlu vektör, Türkçe dahil çok dilli destek
- RAM kullanımı: ~500MB

---

### `RAG/retrieval.py` — Arama ve Filtreleme

```python
class Retriever:
    def search(self, query: str, top_k: int = 8) -> list[Document]:
```

**Çalışma mantığı:**
1. `_normalize_query()`: Türkçe birleşik kelimeleri düzeltir (`"zaman aşımı"` → `"zamanaşımı"`)
2. `similarity_search_with_relevance_scores(k=30)`: 30 aday çeker, cosine similarity skoru hesaplar
3. Skor < 0.30 olan chunk'ları filtreler (alakasız kanunların karışmasını önler)
4. MMR retriever üzerinden çeşitlilik filtresi uygular (`fetch_k=25 → k=8`)
5. Sonuçta en fazla `top_k` döner

**MMR parametreleri:**
- `k=8`: Kaç sonuç döneceği
- `fetch_k=25`: MMR için kaç aday çekileceği
- `lambda_mult=0.6`: Çeşitlilik/alaka dengesi (1.0 = sadece alaka, 0.0 = sadece çeşitlilik)

**`CHROMA_DIR` path hesabı:**
```python
# retrieval.py → RAG/ klasöründe → parent.parent = proje kökü
CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")
```

---

### `backend/main.py` — FastAPI Uygulaması

**Endpoint'ler:**
- `POST /chat` — Ana sohbet endpoint'i
- `DELETE /history/{session_id}` — Oturum geçmişini temizler
- `POST /debug/grounding` — Retrieval kalitesi raporu
- `GET /` — Durum kontrolü

**`/chat` akışı:**
1. `_condense_query()`: Çok turlu sohbette önceki konuyu yeni soruya dahil eder (< 120 karsa)
2. `retriever.search()`: ChromaDB'den ilgili chunk'ları getirir
3. `build_context()`: Chunk'ları `[KanunAdı — MaddeX]\nMetin` formatında birleştirir, aynı içerik tekrar etmez
4. LM Studio'ya gönderir: `SISTEM_PROMPT + geçmiş + BAĞLAM + SORU`
5. `clean_reply()`: `<think>...</think>` bloklarını ve Gemma artifact'larını temizler
6. `filter_grounded_sources()`: Cevap metninde geçen madde numaralarıyla eşleşen kaynakları döner

**`build_context()` dedup anahtarı:**
```python
key = (kanun_adi, madde, txt[:50])
# Aynı maddeden farklı fıkralar ayrı gelir (txt[:50] farkeder)
# Aynı chunk iki kez gelmez
```

**Sistem Promptu kuralları:**
- Yalnızca bağlamdaki metni kullan, uydurma
- Sayısal değerleri (süre, yıl, para) bağlamda yoksa kesinlikle yazma
- Kanunun tam adını ve madde numarasını her atıfta belirt
- `sayılmaz`, `hariç`, `yasaktır` gibi olumsuzlukları tersine çevirme
- Bakanlık → hangi bakanlık olduğunu tam adıyla yaz

---

### `.env` — Ortam Değişkenleri

```env
LM_STUDIO_URL=http://192.168.56.1:1234/v1   # LM Studio sunucu adresi
LM_STUDIO_MODEL=qwen/qwen2.5-7b-instruct-1m # Aktif model adı
COLLECTION_NAME=mevzuat
RETRIEVAL_TOP_K=10
GROQ_API_KEY=...  # Gelecekte Groq API için (şu an kullanılmıyor)
```

**LM Studio IP adresi:** Windows host IP'si (WSL2 veya VM içinden erişim için).
Makine yeniden başlayınca IP değişebilir → `ipconfig` ile kontrol et.

---

### `docker-compose.yml` ve `Dockerfile`

**Qdrant tamamen kaldırıldı**, sadece backend container var.

```yaml
services:
  backend:
    build: .
    ports: ["8000:8000"]
    volumes:
      - .:/app
      - chroma_data:/app/chroma_db  # ChromaDB kalıcı volume
```

```dockerfile
FROM python:3.11-slim
# CPU-only torch (CUDA yok), ChromaDB için sqlite3, pymupdf için gcc
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Not:** PDF ingestion Docker içinde değil, dışarıda çalıştırılmalı.
Çünkü `PDF_DIR` Windows path'i (`C:\Users\elmaz\Desktop\...`).

---

## Başlatma Sırası

### 1. PDF'leri İşle ve ChromaDB'ye Yükle (ilk kurulumda bir kez)
```bash
# venv aktif olmalı
.venv\Scripts\activate

# --reset: mevcut koleksiyonu sil, temiz yükle
python -m RAG.data_ingestion.ingest --reset
```

### 2. Backend'i Başlat
```bash
uvicorn backend.main:app --reload
```

### 3. Test Et
```
http://localhost:8000/docs   ← Swagger UI
http://localhost:8000/       ← Durum
```

---

## Bilinen Sorunlar ve Geçmiş Kararlar

### Neden Qdrant değil ChromaDB?
Qdrant Docker gerektiriyordu ve ilk kurulumda ingestion çok yavaş çalışıyordu.
ChromaDB dosya tabanlı (`chroma_db/` klasörü), Docker olmadan çalışıyor.

### Neden BGE-M3 değil E5-small?
BGE-M3: 2.27GB RAM, reranker ayrıca 600MB daha.
E5-small: ~120MB, 384dim, Türkçe için yeterince iyi.
RAM kısıtı (16GB, LM Studio 6-8GB alıyor) nedeniyle E5-small tercih edildi.

### Neden MultiQueryRetriever kaldırıldı?
`langchain.retrievers` modülü kurulu LangChain versiyonunda `ModuleNotFoundError` verdi.
MMR tek başına zaten çeşitlilik sağlıyor, ayrıca LLM çağrısı gerektirmeden.

### Neden `_split_by_madde()`?
`pymupdf4llm` madde başlıklarını `**Madde 4**` (bold) üretiyor, `## Madde 4` (heading) değil.
`MarkdownHeaderTextSplitter` bunları göremediği için tüm maddeler "Giriş" olarak işaretleniyordu.
Regex-based splitter her "Madde X" ifadesinde keserek bu sorunu çözdü.

### Chunk boyutu neden 1400?
Madde 4 (Güvenlik Soruşturması Kanunu) gibi bullet listesi olan maddeler 900 karakterde bölünüyordu.
LLM eksik parçayı kendisi dolduruyordu (halüsinasyon).
1400 ile bullet listeli maddeler tek chunk olarak kalıyor.

### `score_threshold` neden MMR'dan ayrı?
ChromaDB'nin MMR implementasyonu `score_threshold` parametresini sessizce görmezden geliyor.
Çözüm: `similarity_search_with_relevance_scores()` ile skor filtresi, ardından MMR sıralama.

---

## Chunk Metadata Yapısı

Her ChromaDB chunk'ı şu metadata alanlarını taşır:

| Alan | Örnek | Açıklama |
|------|-------|----------|
| `kanun_adi` | `"gelirvergisi"` | PDF dosya adı (stem) |
| `kanun_tam` | `"Gelir Vergisi Kanunu (193)"` | LLM'in göreceği tam ad |
| `madde` | `"Madde 4"` | Regex ile ayıklanan madde no |
| `kaynak` | `"gelirvergisi.pdf"` | PDF dosya adı |

---

## Mevcut PDF'ler

| Dosya | Kanun |
|-------|-------|
| askerealma.pdf | Askerlik Kanunu (1111) |
| gelirvergisi.pdf | Gelir Vergisi Kanunu (193) |
| guvenliksorusturma.pdf | Güvenlik Soruşturması ve Arşiv Araştırması Kanunu (7315) |
| iskanunu.pdf | İş Kanunu (4857) |
| nufushizmet.pdf | Nüfus Hizmetleri Kanunu (5490) |
| sosyalsigorta_sagliksigorta.pdf | Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu (5510) |
| tuketicihaklari.pdf | Tüketicinin Korunması Hakkında Kanun (6502) |
| turkborclarkanunu.pdf | Türk Borçlar Kanunu (6098) |

---

## Yeni PDF Eklemek

1. PDF'yi `C:\Users\elmaz\Desktop\mevzautlar_kanunlar\` klasörüne koy
2. `ingest.py` içindeki `KANUN_ADLARI` dict'ine ekle:
   ```python
   "yenikanun": "Yeni Kanun Adı (XXXX)",
   ```
3. Yeniden ingest et (mevcut veriye ekler, sıfırlamaz):
   ```bash
   python -m RAG.data_ingestion.ingest
   ```

---

## Önemli Notlar

- **LM Studio açık ve model yüklü olmalı** — backend başlarken `Retriever()` oluşturulur, bu esnada LM Studio'ya bağlantı gerekmez ama sorgu sırasında gerekir.
- **`RETRIEVAL_TOP_K=8`** — `.env`'deki 10 değeri override'dır ama `retriever.k=8` ile hizalanmış olmalı.
- **`chroma_db/` klasörü** — Git'e commit edilmemeli (büyük), ama silinirse `ingest --reset` ile yeniden oluşturulabilir.
- **RAM:** LM Studio 7B model 6-8GB, E5-small 500MB, Windows 4-5GB alıyor. 16GB RAM'de dar.
  LM Studio'da modeli kullanmadığında "Eject" et.

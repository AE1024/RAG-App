# Türkiye Cumhuriyeti Kanunları RAG Chatbot

Türk kanun metinlerini (PDF) işleyip hukuki sorulara **tamamen yerel** cevap veren bir Retrieval-Augmented Generation sistemi. İnternet bağlantısı gerektirmez, veriler dışarı çıkmaz.

---

## İçindekiler

- [Proje Özeti](#proje-özeti)
- [Mimari](#mimari)
- [Klasör Yapısı](#klasör-yapısı)
- [Kurulum](#kurulum)
- [Başlatma Sırası](#başlatma-sırası)
- [Veri Yükleme Pipeline'ı](#veri-yükleme-pipelineı)
- [Retrieval Pipeline'ı](#retrieval-pipelineı)
- [Guardrail Sistemi](#guardrail-sistemi)
- [Gözlemlenebilirlik](#gözlemlenebilirlik)
- [API Referansı](#api-referansı)
- [Test](#test)
- [Desteklenen Kanunlar](#desteklenen-kanunlar)
- [Konfigürasyon](#konfigürasyon)
- [Bilinen Kısıtlar](#bilinen-kısıtlar)

---

## Proje Özeti

| Bileşen | Teknoloji |
|---------|-----------|
| Embedding modeli | `intfloat/multilingual-e5-small` (120 MB, 384 dim) |
| Vektör veritabanı | ChromaDB (yerel dosya, Docker gerektirmez) |
| Üretici LLM | LM Studio — Qwen2.5-7B (tamamen yerel) |
| Guardrail LLM | Groq — `openai/gpt-oss-120b` (konu sınıflandırma + grounding skoru) |
| API katmanı | FastAPI |
| Gözlemlenebilirlik | LangSmith tracing + JSONL logging |
| RAG değerlendirme | DeepEval (Faithfulness, Hallucination, GEval…) |

---

## Mimari

```
Kullanıcı Sorusu
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                   GUARDRAIL — Katman 1                  │
│  LengthGuard (max 2000 kar)  •  PIIGuard (KVKK)        │
└───────────────────────┬─────────────────────────────────┘
                        │ geçti
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   GUARDRAIL — Katman 2                  │
│        Groq: konu sınıflandırması (allowed / not)      │
└───────────────────────┬─────────────────────────────────┘
                        │ allowed
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  RETRIEVAL (ChromaDB)                   │
│  query → "query: " prefix → E5Small embed              │
│  child chunk'lar arasında similarity search (k=60)     │
│  score < 0.52 → elenir                                 │
│  parent_id → parents.json'dan tam madde metni          │
│  top-k (varsayılan 4) parent döner                     │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                GUARDRAIL — Katman 1                     │
│  RetrievalEmptyGuard • IndirectInjectionSanitizer      │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              LM Studio / Qwen2.5-7B (yerel)             │
│  SISTEM PROMPTU + BAĞLAM (maddeler) + SORU             │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                GUARDRAIL — Katman 2                     │
│        Groq: grounding skoru (1-5 hallucination)       │
│        Skor ≥ 4 → yanıta uyarı eklenir                 │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
             JSON Response + Kaynaklar
```

---

## Klasör Yapısı

```
llm_proje/
│
├── RAG/
│   ├── embedding.py              # E5SmallEmbeddings — passage:/query: prefix yönetimi
│   ├── retrieval.py              # ChromaDB parent-child retriever
│   └── data_ingestion/
│       ├── ingest.py             # PDF → ChromaDB pipeline (çalıştırılacak komut)
│       └── markdowns/            # PDF'ten üretilen .md dosyaları buraya gider
│
├── backend/
│   ├── main.py                   # FastAPI uygulaması, /chat endpoint
│   ├── tracing.py                # LangSmith tracing sarmalayıcısı
│   ├── logger.py                 # Structured JSONL logger
│   └── guardrails/
│       ├── groq_client.py        # Async Groq istemcisi
│       ├── forward_guards.py     # Katman-1: regex/stdlib guard'lar
│       ├── input_guards.py       # Katman-2: konu sınıflandırma (Groq)
│       └── output_guards.py      # Katman-2: grounding skoru (Groq)
│
├── backend/tests/
│   ├── judge.py                  # DeepEval Groq judge modeli
│   ├── tests_rag.py              # DeepEval RAG kalite testleri
│   ├── conftest.py               # Pytest mock kurulumu
│   └── unit_tests/
│       └── test_helpers.py       # Saf fonksiyon unit testleri
│
├── chroma_db/                    # ChromaDB verileri (otomatik oluşur, git'e eklenmez)
│   └── parents.json              # Tam madde metinleri (parent store)
│
├── logs/
│   └── chat.jsonl                # Structured request logları
│
├── conftest.py                   # Pytest sys.path kurulumu
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env
```

---

## Kurulum

### Gereksinimler

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) (yerel LLM sunucusu)
- Groq API anahtarı (guardrail + test judge için)

### 1. Sanal ortam ve bağımlılıklar

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Ortam değişkenleri

`.env` dosyasını oluşturun:

```env
# LM Studio — Windows host IP'si (ipconfig ile bulun)
LM_STUDIO_URL=http://192.168.X.X:1234/v1
LM_STUDIO_MODEL=qwen/qwen2.5-7b-instruct-1m

COLLECTION_NAME=mevzuat
RETRIEVAL_TOP_K=4

# Guardrail + test judge
GROQ_API_KEY=gsk_...

# LangSmith (opsiyonel)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=RAG-App
```

> **Not:** `LM_STUDIO_URL` WSL2 veya VM içinden erişiyorsanız Windows host IP'sini kullanır. `ipconfig` ile `Ethernet adapter vEthernet` satırını kontrol edin.

---

## Başlatma Sırası

```
1. LM Studio'yu aç → modeli yükle → sunucuyu başlat
2. PDF'leri işle (ilk kurulumda bir kez)
3. Backend'i başlat
```

### Adım 2 — PDF'leri ChromaDB'ye yükle

```bash
# Sıfırdan yükle (mevcut veriyi siler)
python -m RAG.data_ingestion.ingest --reset

# Sadece yeni PDF ekle (mevcut veriye dokunmaz)
python -m RAG.data_ingestion.ingest
```

### Adım 3 — Backend'i başlat

```bash
uvicorn backend.main:app --reload
```

Swagger UI: `http://localhost:8000/docs`

---

## Veri Yükleme Pipeline'ı

PDF'ler önce Markdown'a çevrilip `RAG/data_ingestion/markdowns/` altına kaydedilir, ardından ingest pipeline'ı çalışır.

```
PDF
 │
 ▼
pymupdf4llm → Markdown
 │
 ▼  _clean_markdown()
 │   • PDF başlık bloğunu sil ("Kanun Numarası : X ...")
 │   • Değişiklik cetvelini sil
 │   • Mülga maddeleri stub içerikle koru
 │   • "(Değişik: ...)" satır içi notlarını sil
 │
 ▼  _split_into_parents()
 │   Regex: her "Madde X" ifadesinde kes
 │   → Her madde = 1 Parent Document (tam metin)
 │   Metadata: kanun_adi, kanun_tam, madde, kaynak
 │
 ▼  _make_children()
 │   Her parent'ı ~400 karakterlik child chunk'lara böl
 │   Devam chunk'larına "[Madde X — devam N/M]" başlığı ekle
 │   Her child, parent_id taşır
 │
 ▼
ChromaDB (child chunk'lar — embed edilir)
 +
parents.json (parent tam metinleri — ID ile saklanır)
```

### Neden Parent-Child stratejisi?

Embedding kalitesi için **küçük** (400 kar) chunk'lar kullanılır; ama LLM'e **tam madde metni** gönderilir. Küçük chunk doğru maddeyi bulur, büyük parent metin tam bağlamı sağlar.

### Neden `_split_into_parents()` custom regex?

`pymupdf4llm`, madde başlıklarını `**Madde 4**` (bold) olarak üretir, `## Madde 4` (heading) olarak değil. LangChain'in `MarkdownHeaderTextSplitter`'ı bu formatı göremez. Regex-based splitter bu sorunu çözer.

---

## Retrieval Pipeline'ı

```
Kullanıcı sorusu
        │
        ▼
_normalize_query()
  "zaman aşımı" → "zamanaşımı"  (Türkçe birleşik kelime düzeltmeleri)
        │
        ▼
"query: " + sorgu  →  E5Small embed
        │
        ▼
ChromaDB similarity search
  k=60, filter={"doc_type": "child"}  (sadece child'lar arasında ara)
        │
        ▼
Score filtresi: score < 0.52 → elenir
(Hiçbiri geçmezse en iyi top_k ile devam)
        │
        ▼
Parent dedup: her parent'ın en yüksek child skorunu tut
        │
        ▼
parents.json → tam madde metinleri
        │
        ▼
top-k (varsayılan 4) Document döner
```

### Embedding modeli — E5Small

`intfloat/multilingual-e5-small` modeli **prefix** gerektirir:

| Kullanım | Prefix |
|----------|--------|
| Belge embed (ingest) | `"passage: "` + metin |
| Sorgu embed (arama) | `"query: "` + sorgu |

Prefix olmadan vektörler farklı uzayda olur, benzerlik skoru düşer.

---

## Guardrail Sistemi

İki katmanlı koruma sistemi, her chat isteğinde sırayla çalışır.

### Katman 1 — Hızlı (LLM çağrısı yok)

```
1. LengthGuard      Max 2000 karakter. Aşılırsa → 400 HTTP hatası
2. PIIGuard         TC kimlik, telefon, IBAN tespiti → uyarı (bloklamaz)
3. RetrievalEmpty   Retriever 0 sonuç döndürdüyse → LLM'e gönderme
4. InjectionSan.    Retrieved chunk'lardaki "ignore instructions" kalıplarını temizle
```

### Katman 2 — Groq (LLM-as-judge)

```
5. input_guardrail   Soru Türk hukukuyla ilgili mi? → "allowed" / "not_allowed"
6. output_guardrail  Yanıt bağlamla örtüşüyor mu? → 1-5 hallucination skoru
                     Skor ≥ 4 → yanıta kullanıcı uyarısı eklenir
```

### Guard akışı (main.py sırası)

```
pre_retrieval()   [Katman 1 — Retriever'dan ÖNCE]
   └─ LengthGuard + PIIGuard

input_guardrail() [Katman 2 — Retriever'dan ÖNCE]
   └─ Groq konu sınıflandırması

retriever.search()

pre_llm()         [Katman 1 — LLM'den ÖNCE]
   └─ RetrievalEmptyGuard + IndirectInjectionSanitizer

LLM çağrısı

output_guardrail() [Katman 2 — LLM'den SONRA]
   └─ Groq grounding skoru
```

---

## Gözlemlenebilirlik

### LangSmith Tracing

Her `/chat` isteği LangSmith'e izleme verisi gönderir. `.env`'deki `LANGSMITH_TRACING=true` ile aktif olur; set edilmezse otomatik no-op çalışır.

İzlenen bileşenler:
- `chat_pipeline` — tüm uçtan uca akış
- `retrieval` — ChromaDB sorgusu ve skoru

`LANGSMITH_PROJECT=RAG-App` ile LangSmith arayüzünde proje altında gruplanır.

### JSONL Logging

Her istek `logs/chat.jsonl` dosyasına bir JSON satırı olarak yazılır:

```json
{
  "ts": "2026-08-19T10:23:01Z",
  "request_id": "uuid",
  "session_id": "default",
  "query_masked": "kira sözleşmesi...",
  "guard_pre": "warn:pii",
  "guard_topic": "pass",
  "guard_empty": "pass",
  "groq_grounding": 2,
  "retrieval_count": 4,
  "sources_count": 2,
  "tokens_prompt": 1240,
  "tokens_completion": 180,
  "latency_ms": 3420,
  "early_exit": null
}
```

PII içeren sorgular `PIIGuard.mask()` ile maskelenerek loglanır (`query_masked`).

---

## API Referansı

### `POST /chat`

```json
// İstek
{
  "message": "Kira sözleşmesinde depozito kaç ay olabilir?",
  "session_id": "user-123"
}

// Yanıt
{
  "response": "Türk Borçlar Kanunu Madde 342'ye göre...",
  "sources": [
    {
      "law_name": "Türk Borçlar Kanunu (6098)",
      "law": "Türk Borçlar Kanunu (6098)",
      "source": "turkborclarkanunu.pdf"
    }
  ]
}
```

### `POST /debug/grounding`

Retrieval kalitesini raporlar — hangi kaynaklar kullanıldı, hangisi kullanılmadı.

```json
// İstek
{"question": "ihbar süresi nedir?", "response": "iki haftadır"}

// Yanıt
{
  "retrieved_count": 4,
  "grounded_count": 1,
  "grounding_ratio": 0.25,
  "grounded_sources": [...],
  "unused_sources": [...]
}
```

### `GET /`

Sistem durum kontrolü: `{"status": "ok", "model": "...", "collection": "mevzuat"}`

---

## Test

### DeepEval RAG Kalite Testleri

Groq (`openai/gpt-oss-120b`) judge modeli kullanarak RAG pipeline kalitesini değerlendirir. LM Studio veya ChromaDB **gerektirmez** — test case'ler statik.

```bash
pytest backend/tests/tests_rag.py -v
```

| Test | Metrik | Ne test eder |
|------|--------|--------------|
| `test_faithfulness_*` | FaithfulnessMetric | Yanıt bağlama sadık mı? |
| `test_answer_relevancy_*` | AnswerRelevancyMetric | Yanıt soruyla ilgili mi? |
| `test_hallucination_*` | HallucinationMetric | Bağlamla çelişen bilgi var mı? |
| `test_contextual_precision_*` | ContextualPrecisionMetric | Alakalı chunk'lar önde mi? |
| `test_contextual_recall_*` | ContextualRecallMetric | Bağlam yeterince kapsamlı mı? |
| `test_contextual_relevancy_*` | ContextualRelevancyMetric | Chunk'lar soruyla ilgili mi? |
| `test_geval_*` | GEval | Kanun atıfı, feragatname, olumsuzluk koruması |

### Unit Testler

LLM, API veya ChromaDB gerektirmez. Saf yardımcı fonksiyonları test eder.

```bash
pytest backend/tests/unit_tests/ -v
```

| Test grubu | Fonksiyon | Örnekler |
|------------|-----------|----------|
| `test_normalize_*` | `_normalize_query` | "zaman aşımı" → "zamanaşımı" |
| `test_clean_reply_*` | `clean_reply` | `<think>` bloğu kaldırma |
| `test_filter_*` | `filter_grounded_sources` | "bilgi bulunamadı" → boş liste |
| `test_build_context_*` | `build_context` | Dedup, truncation, format |

```bash
# Tüm testler
pytest backend/tests/ -v
```

---

## Desteklenen Kanunlar

| PDF dosyası | Kanun |
|-------------|-------|
| askerealma.pdf | Askerlik Kanunu (1111) |
| gelirvergisi.pdf | Gelir Vergisi Kanunu (193) |
| guvenliksorusturma.pdf | Güvenlik Soruşturması ve Arşiv Araştırması Kanunu (7315) |
| iskanunu.pdf | İş Kanunu (4857) |
| nufushizmet.pdf | Nüfus Hizmetleri Kanunu (5490) |
| sosyalsigorta_sagliksigorta.pdf | Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu (5510) |
| tuketicihaklari.pdf | Tüketicinin Korunması Hakkında Kanun (6502) |
| turkborclarkanunu.pdf | Türk Borçlar Kanunu (6098) |

### Yeni Kanun Eklemek

1. PDF'yi `RAG/data_ingestion/` dizinine koy (veya pdf_to_markdown scriptiyle .md üret)
2. `ingest.py` içindeki `KANUN_ADLARI` dict'ine ekle:
   ```python
   "yenidosyaadi": "Yeni Kanun Adı (XXXX)",
   ```
3. Yeniden ingest et:
   ```bash
   python -m RAG.data_ingestion.ingest
   ```

---

## Konfigürasyon

| Değişken | Varsayılan | Açıklama |
|----------|-----------|---------|
| `LM_STUDIO_URL` | — | LM Studio OpenAI-compat endpoint |
| `LM_STUDIO_MODEL` | — | Aktif model adı |
| `RETRIEVAL_TOP_K` | `4` | Retriever'dan kaç parent dönsün |
| `GROQ_API_KEY` | — | Guardrail + DeepEval judge için |
| `LANGSMITH_TRACING` | — | `true` yazılırsa tracing aktif |
| `LANGSMITH_API_KEY` | — | LangSmith erişim anahtarı |
| `LANGSMITH_PROJECT` | — | LangSmith proje adı |
| `COLLECTION_NAME` | `mevzuat` | ChromaDB koleksiyon adı |

### Temel parametreler (kod içi)

| Parametre | Değer | Konum |
|-----------|-------|-------|
| Score threshold | 0.52 | `retrieval.py` |
| Child chunk boyutu | 400 kar | `ingest.py` |
| Max context karakter | 3000 | `main.py` |
| Max sorgu uzunluğu | 2000 kar | `forward_guards.py` |
| LLM sıcaklığı | 0.4 | `main.py` |

---

## Bilinen Kısıtlar

**RAM kullanımı** — LM Studio 7B model ~6-8 GB, E5-small ~500 MB, Windows ~4-5 GB alır. 16 GB RAM'de sıkışık çalışır. LM Studio'da modeli kullanmadığınızda "Eject" edin.

**LM Studio IP adresi** — Windows makinesi yeniden başlayınca WSL2 veya VM'in gördüğü host IP değişebilir. `.env`'deki `LM_STUDIO_URL`'yi `ipconfig` ile güncel IP ile güncelleyin.

**PDF ingestion Docker içinde çalışmaz** — `PDF_DIR` Windows path'i olduğundan ingestion doğrudan host üzerinde çalıştırılmalıdır.

**`filter_grounded_sources` kaynak eşleştirmesi** — Fonksiyon `src.law` içindeki ilk sayıyı madde numarasıyla karşılaştırır. Gerçek kayıtlarda kanun numarası (4857) madde numarasından önce geldiğinden eşleşme sağlanamamakta, tüm kaynaklar fallback olarak dönmektedir. "Bilgi bulunamadı" filtrelemesi doğru çalışmaktadır.

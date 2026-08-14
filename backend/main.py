import os
import re
import sys
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RAG.retrieval import Retriever

load_dotenv()

LM_STUDIO_URL   = os.getenv("LM_STUDIO_URL", "http://172.19.64.1:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen2.5-7b-instruct-1m")
MAX_HISTORY     = 6
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))

SYSTEM_PROMPT = """Sen Türk hukuku konusunda uzman bir yapay zeka asistanısın.
Görevin, kullanıcıların Türk kanunları ve yönetmelikleri hakkındaki sorularını yanıtlamaktır.

KURALLAR:
- Yalnızca sana verilen BAĞLAM bölümündeki kanun metinlerini kullanarak cevap ver.
- Bağlamda bulunmayan bilgileri kesinlikle uydurma veya tahmin etme.
- KRİTİK: Süre, yıl, gün, para miktarı gibi sayısal değerleri ASLA bağlamdan bağımsız yazma. Bağlamda "10 yıl" yazmıyorsa "10 yıl" deme. Bağlamda süre yoksa: "Elimdeki bağlamda bu konunun süre bilgisi yer almamaktadır." de.
- Bağlamda birden fazla ilgili madde varsa hepsini değerlendir: genel kural, istisnalar, başlangıç ve durma halleri gibi tüm boyutları kapsayan bir cevap ver.
- Her atıf için kanunun TAM ADINI ve madde numarasını belirt (örn: "Türk Borçlar Kanunu Madde 146").
- TEMEL UYARI — Terim Tutarlılığı: Her kanunda "Bakanlık" farklı bir bakanlığı ifade edebilir.
  Askerlik/savunma kanunlarında "Bakanlık" = Milli Savunma Bakanlığı'dır.
  Sağlık kanunlarında "Bakanlık" = Sağlık Bakanlığı'dır.
  Cevabında daima tam adı yaz, asla yalnızca "Bakanlık" yazma.
- Aynı bilgiyi birden fazla kez tekrarlama; her noktayı yalnızca bir kez açıkla.
- Bağlamda soruyla ilgili bilgi yoksa "Elimdeki kanun metinlerinde bu konuya dair bilgi bulunamadı." de.
- Cevaplarını her zaman Türkçe ver.
- Hukuki tavsiye vermediğini, yalnızca kanun metinlerini aktardığını belirt."""

# ---------------------------------------------------------------------------
app = FastAPI(title="RAG Hukuk Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
retriever  = Retriever()

conversation_history: dict[str, list] = defaultdict(list)

# ---------------------------------------------------------------------------
# Modeller
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message:    str
    session_id: str = "default"

class Source(BaseModel):
    kanun_adi: str
    madde:str
    kaynak:    str

class ChatResponse(BaseModel):
    response: str
    sources:  list[Source]

_THINK_RE    = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_ARTIFACT_RE = re.compile(r'^\s*\[\s*\]\s*:\s*', re.MULTILINE)  # Gemma "[]: " artifact

def clean_reply(text: str) -> str:
    text = _THINK_RE.sub('', text)
    text = _ARTIFACT_RE.sub('', text)
    return text.strip()


def build_context(results: list) -> tuple[str, list[Source]]:
    # LangChain Document: .page_content (metin), .metadata (kanun_adi, madde, kaynak)
    # metin[:50] ile aynı maddenin farklı fıkra chunk'ları ayrı ayrı LLM'e gider.
    seen:    set[tuple[str, str, str]] = set()
    parts:   list[str]   = []
    sources: list[Source] = []

    for doc in results:
        m   = doc.metadata
        txt = doc.page_content
        key = (m.get('kanun_adi', ''), m.get('madde', ''), txt[:50])
        if key in seen:
            continue
        seen.add(key)
        kanun_label = m.get('kanun_tam') or m.get('kanun_adi', '')
        parts.append(f"[{kanun_label} — {m.get('madde', '')}]\n{txt}")
        sources.append(Source(
            kanun_adi=kanun_label,
            madde=m.get('madde', ''),
            kaynak=m.get('kaynak', ''),
        ))

    return "\n\n".join(parts), sources


_CONDENSE_SYSTEM = (
    "Türkçe hukuk bilgi tabanında arama yapılacak. "
    "Sohbet geçmişini ve yeni soruyu inceleyerek, "
    "tek başına anlaşılır ve eksiksiz bir arama sorgusu oluştur. "
    "Yalnızca sorgu metnini döndür, başka hiçbir şey yazma."
)


def _condense_query(message: str, history: list) -> str:
    """Multi-turn konuşmada retrieval için bağımsız sorgu üretir.
    Kısa veya belirsiz sorularda geçmişteki konuyu retrieval'e taşır."""
    # Mesaj yeterince uzunsa ya da geçmiş yoksa doğrudan kullan
    if not history or len(message) > 120:
        return message

    recent = history[-4:]  # son 2 tur
    history_str = "\n".join(
        f"{'Kullanıcı' if m['role'] == 'user' else 'Asistan'}: {m['content'][:300]}"
        for m in recent
    )
    try:
        r = llm_client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=[
                {"role": "system",  "content": _CONDENSE_SYSTEM},
                {"role": "user",    "content": f"Geçmiş:\n{history_str}\n\nYeni soru: {message}"},
            ],
            temperature=0.2,
            max_tokens=120,
        )
        condensed = clean_reply(r.choices[0].message.content or "")
        return condensed if condensed else message
    except Exception:
        return message  # LLM hatası → orijinal soruyla devam et


def filter_grounded_sources(reply: str, sources: list[Source]) -> list[Source]:
    """Cevap metninde gerçekten atıf yapılan kaynakları döndürür."""
    mentioned = set(re.findall(r'(?:[Mm]adde\s*|m\.\s*)(\d+)', reply))
    grounded  = [
        src for src in sources
        if (m := re.search(r'(\d+)', src.madde)) and m.group(1) in mentioned
    ]
    return grounded if grounded else sources

# ---------------------------------------------------------------------------
# Endpoint'ler
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        history = conversation_history[request.session_id]

        # 1. Multi-turn için retrieval sorgusunu bağımsız hale getir
        search_query = _condense_query(request.message, history)

        # 2. Dense retrieval + cross-encoder reranking
        results  = retriever.search(search_query, top_k=RETRIEVAL_TOP_K)
        context, sources = build_context(results)

        # 3. LLM'e ORIJINAL mesajı gönder (condensed değil); bağlamı ekle
        user_content = f"BAĞLAM:\n{context}\n\nSORU: {request.message}"
        history.append({"role": "user", "content": user_content})

        # 3. Lokal LLM'e gönder
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-MAX_HISTORY:]
        response = llm_client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=messages,
            temperature=0.4,
        )
        msg       = response.choices[0].message
        # Qwen3 thinking modu: content boşsa reasoning_content'e fallback
        raw_reply = msg.content or getattr(msg, "reasoning_content", None) or ""
        reply     = clean_reply(raw_reply)

        # 4. Geçmişe kaydet
        history.append({"role": "assistant", "content": reply})

        return ChatResponse(
            response=reply,
            sources=filter_grounded_sources(reply, sources),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    conversation_history.pop(session_id, None)
    return {"message": f"'{session_id}' geçmişi temizlendi."}


@app.post("/debug/grounding")
def grounding_check(req: dict):
    """Test amaçlı: retrieval + grounding raporu."""
    question = req.get("question", "")
    response = req.get("response", "")
    results, all_sources = build_context(retriever.search(question, top_k=5))
    grounded = filter_grounded_sources(response, all_sources)
    unused   = [s for s in all_sources if s not in grounded]
    return {
        "retrieved_count":  len(all_sources),
        "grounded_count":   len(grounded),
        "grounding_ratio":  round(len(grounded) / len(all_sources), 2) if all_sources else 0.0,
        "think_present":    bool(_THINK_RE.search(response)),
        "grounded_sources": [s.model_dump() for s in grounded],
        "unused_sources":   [s.model_dump() for s in unused],
    }


@app.get("/")
def root():
    return {"status": "ok", "model": LM_STUDIO_MODEL, "collection": "mevzuat"}

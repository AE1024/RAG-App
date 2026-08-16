import os
import re
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

from backend.guardrails.forward_guards import ForwardGuardPipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RAG.retrieval import Retriever

load_dotenv()

LM_STUDIO_URL   = os.getenv("LM_STUDIO_URL", "http://172.19.64.1:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen2.5-7b-instruct-1m")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

SYSTEM_PROMPT = """Sen Türk hukuku konusunda uzman bir yapay zeka asistanısın.
Görevin, kullanıcıların Türk kanunları ve yönetmelikleri hakkındaki sorularını yanıtlamaktır.

KURALLAR:
- Yalnızca sana verilen BAĞLAM bölümündeki kanun metinlerini kullanarak cevap ver.
- Bağlamda bulunmayan bilgileri kesinlikle uydurma veya tahmin etme.
- KRİTİK: Süre, yıl, gün, para miktarı gibi sayısal değerleri ASLA bağlamdan bağımsız yazma. Bağlamda "10 yıl" yazmıyorsa "10 yıl" deme. Bağlamda süre yoksa "Elimdeki bağlamda bu konunun süre bilgisi yer almamaktadır." de.
- Her atıf için kanunun TAM ADINI ve madde numarasını belirt (örn: "Türk Borçlar Kanunu Madde 146").
- TEMEL UYARI — Terim Tutarlılığı: Her kanunda "Bakanlık" farklı bir bakanlığı ifade edebilir.
  Askerlik/savunma kanunlarında "Bakanlık" = Milli Savunma Bakanlığı'dır.
  Sağlık kanunlarında "Bakanlık" = Sağlık Bakanlığı'dır.
  Cevabında daima tam adı yaz, asla yalnızca "Bakanlık" yazma.
- Bağlamda soruyla ilgili bilgi yoksa "Elimdeki kanun metinlerinde bu konuya dair bilgi bulunamadı." de.
- Bağlamda verilen bir madde soruyla doğrudan ilgili değilse o maddeyi cevaba dahil etme.
- Kanun metnindeki 'sayılmaz', 'hariç', 'yasaktır' gibi istisna veya olumsuzluk bildiren kelimeleri çok dikkatli oku, anlamı tersine çevirme.
- Cevabını kısa ve odaklı tut. Soruya direkt yanıt ver, ilgisiz detayları sıralama.
- Hukuki tavsiye vermediğini, yalnızca kanun metinlerini aktardığını belirt.
- Cevaplarını her zaman Türkçe ver.
"""

app = FastAPI(title="RAG Hukuk Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
retriever  = Retriever()
forward_guard_pipeline = ForwardGuardPipeline()

class ChatRequest(BaseModel):
    message:    str
    session_id: str = "default"

class Source(BaseModel):
    law_name: str
    law:     str
    source:    str

class ChatResponse(BaseModel):
    response: str
    sources:  list[Source]


_THINK_RE    = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_ARTIFACT_RE = re.compile(r'^\s*\[\s*\]\s*:\s*', re.MULTILINE)

def clean_reply(text: str) -> str:
    text = _THINK_RE.sub('', text)
    text = _ARTIFACT_RE.sub('', text)
    return text.strip()


MAX_PARENT_CHARS = 3000  # Uzun maddeler (Gelir Vergisi muafiyet listeleri vb.) token limitini patlatmasın

def build_context(results: list) -> tuple[str, list[Source]]:
    seen:    set[tuple] = set()
    parts:   list[str]  = []
    sources: list[Source] = []

    for doc in results:
        m   = doc.metadata
        txt = doc.page_content
        key = (m.get('kanun_adi', ''), m.get('madde', ''), txt[:50])
        if key in seen:
            continue
        seen.add(key)
        kanun_label = m.get('kanun_tam') or m.get('kanun_adi', '')
        # Çok uzun maddeler LLM'in context limitini aşmasın
        if len(txt) > MAX_PARENT_CHARS:
            txt = txt[:MAX_PARENT_CHARS] + "\n[...]"
        parts.append(f"[{kanun_label} — {m.get('madde', '')}]\n{txt}")
        sources.append(Source(
            law_name=kanun_label,
            law=m.get('kanun_tam') or m.get('kanun_adi', ''),
            source=m.get('kaynak', ''),
        ))

    return "\n\n".join(parts), sources


_NOT_FOUND_RE = re.compile(r'bilgi bulunamadı|bilgi yer almamaktadır', re.IGNORECASE)

def filter_grounded_sources(reply: str, sources: list[Source]) -> list[Source]:
    # "Bilgi bulunamadı" yanıtında kaynak listesi boş dönsün (10 alakasız madde gösterme)
    if _NOT_FOUND_RE.search(reply):
        return []
    mentioned = set(re.findall(r'(?:[Mm]adde\s*|m\.\s*)(\d+)', reply))
    grounded  = [
        src for src in sources
        if (m := re.search(r'(\d+)', src.law)) and m.group(1) in mentioned
    ]
    return grounded if grounded else sources


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Aşama 1 — retriever'dan önce: LengthGuard + PIIGuard
        pre_results = forward_guard_pipeline.pre_retrieval(request.message)
        blocked = next((r for r in pre_results if not r.passed and r.severity == "block"), None)
        if blocked:
            raise HTTPException(status_code=400, detail=blocked.reason)

        # Retrieval
        results = retriever.search(request.message, top_k=RETRIEVAL_TOP_K)
        context, sources = build_context(results)

        # Aşama 2 — LLM'den önce: RetrievalEmptyGuard + IndirectInjectionSanitizer
        empty_guard, context = forward_guard_pipeline.pre_llm(results, context)
        if not empty_guard.passed:
            return ChatResponse(response=str(empty_guard.reason), sources=[])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"BAĞLAM:\n{context}\n\nSORU: {request.message}"},
        ]
        response = llm_client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=messages,
            temperature=0.4,
        )
        msg= response.choices[0].message
        raw_reply = msg.content or getattr(msg, "reasoning_content", None) or ""
        reply= clean_reply(raw_reply)

        # Aşama 3 — LLM'den sonra: NumericalHallucinationGuard
        post_results = forward_guard_pipeline.post_llm(reply, context)
        warn = next((r for r in post_results if not r.passed and r.severity == "warn"), None)
        if warn:
            reply = reply + f"\n\n {warn.reason}"

        return ChatResponse(
            response=reply,
            sources=filter_grounded_sources(reply, sources),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/grounding")
def grounding_check(req: dict):
    question = req.get("question", "")
    response = req.get("response", "")
    _, all_sources = build_context(retriever.search(question, top_k=4))
    grounded = filter_grounded_sources(response, all_sources)
    unused   = [s for s in all_sources if s not in grounded]
    return {
        "retrieved_count":  len(all_sources),
        "grounded_count":   len(grounded),
        "grounding_ratio":  round(len(grounded) / len(all_sources), 2) if all_sources else 0.0,
        "grounded_sources": [s.model_dump() for s in grounded],
        "unused_sources":   [s.model_dump() for s in unused],
    }


@app.get("/")
def root():
    return {"status": "ok", "model": LM_STUDIO_MODEL, "collection": "mevzuat"}

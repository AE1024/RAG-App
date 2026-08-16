import asyncio

from backend.guardrails.groq_client import Client_Groq
from backend.guardrails.input_guards import execute_chat_with_guardrail

groq_client = Client_Groq()

_BLOCKED_MSG = "Bu sistem yalnızca Türk hukuku ve mevzuatıyla ilgili sorulara yanıt vermektedir."
_HALLUCINATION_MSG = (
    " Bu yanıt, mevcut kanun metinlerinde doğrulanamayan bilgiler içerebilir. "
    "Lütfen bir hukuk uzmanına danışın."
)

_GROUNDING_SYSTEM = (
    "Sen bir hukuki RAG sisteminin çıktısını denetleyen bir kontrol asistanısın.\n\n"
    "Görevin: Verilen YANIT'ın, verilen BAĞLAM içinde yer almayan kanun adı, "
    "madde numarası, süre, para cezası veya kesin hukuki iddia içerip içermediğini "
    "değerlendirmek.\n\n"
    "Değerlendirme adımları:\n"
    "1. BAĞLAM ve YANIT'ı dikkatlice oku.\n"
    "2. YANIT içindeki her somut hukuki iddiayı (kanun adı, madde, süre, ceza) tespit et.\n"
    "3. Bu iddialardan kaçı BAĞLAM'da doğrulanabiliyor, kaçı doğrulanamıyor say.\n"
    "4. 1'den 5'e bir halüsinasyon skoru ver:\n"
    "   1 = tüm iddialar bağlamda mevcut (iyi yanıt)\n"
    "   5 = birden fazla iddia bağlamda hiç yer almıyor (halüsinasyon)\n\n"
    "Sadece rakam yaz (1-5). Başka hiçbir şey yazma."
)


async def moderation_guardrail(context: str, chat_response: str) -> str:
    """Yanıtı bağlamla karşılaştırıp 1-5 arası halüsinasyon skoru döner."""
    print("Grounding guardrail kontrol ediliyor...")
    messages = [
        {"role": "system", "content": _GROUNDING_SYSTEM},
        {
            "role": "user",
            "content": (
                f"### BAĞLAM (retrieved context)\n{context}\n\n"
                f"### YANIT (kontrol edilecek)\n{chat_response}"
            ),
        },
    ]
    score_str = await groq_client.get_chat_response_async(messages, temperature=0)
    print(f"Grounding skoru: {score_str!r}")
    return str(score_str)


async def execute_all_guardrails(user_request: str, context: str = "") -> str:
    """
    Tam guardrail pipeline:
      1. execute_chat_with_guardrail → konu dışıysa erken çık
      2. moderation_guardrail        → halüsinasyon skoru yüksekse uyar
    """
    # Adım 1: Konu kontrolü + chat (execute_chat_with_guardrail içinde input_guardrail var)
    chat_response = await execute_chat_with_guardrail(user_request)

    if chat_response == _BLOCKED_MSG:
        return chat_response

    # Adım 2: Grounding kontrolü
    score_str = await moderation_guardrail(context, str(chat_response))

    try:
        score = int(score_str.strip())
    except ValueError:
        print(f"Geçersiz skor formatı: {score_str!r}, yanıt olduğu gibi döndürülüyor.")
        return str(chat_response)

    if score >= 3:
        print(f"Grounding guardrail devreye girdi, skor: {score}")
        return _HALLUCINATION_MSG

    print(f"Grounding geçildi, skor: {score}")
    return str(chat_response)


if __name__ == "__main__":
    async def _test():
        test_context = (
            "[İş Kanunu (4857) — Madde 17]\n"
            "Belirsiz süreli iş sözleşmelerinin feshinden önce durumun diğer "
            "tarafa bildirilmesi gerekir. İş sözleşmeleri; işi altı aydan az "
            "sürmüş işçi için iki hafta sonra feshedilmiş sayılır."
        )
        good = "İş sözleşmesinde ihbar süresi ne kadardır?"
        bad = "Bugün İstanbul'da hava nasıl?"

        print("--- Alakasız soru (konu guardrail bekleniyor) ---")
        print(await execute_all_guardrails(bad, context=""))

        print("\n--- Hukuki soru (grounding kontrol bekleniyor) ---")
        print(await execute_all_guardrails(good, context=test_context))

    asyncio.run(_test())

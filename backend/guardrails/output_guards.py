from backend.guardrails.groq_client import Client_Groq

groq_client = Client_Groq()

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


async def output_guardrail(context: str, chat_response: str) -> str:
    """Yanıtı bağlamla karşılaştırıp 1-5 arası halüsinasyon skoru döner.
    1-2: iyi, 3: orta, 4-5: uyarı eşiği (main.py'de >= 4 kullanıcıya uyarı eklenir).
    """
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
    result = await groq_client.get_chat_response_async(messages, temperature=0)
    result = str(result).strip()
    return result.splitlines()[0]  


if __name__ == "__main__":
    import asyncio

    async def _test():
        context = (
            "[İş Kanunu (4857) — Madde 17]\n"
            "İş sözleşmeleri; işi altı aydan az sürmüş işçi için iki hafta "
            "sonra feshedilmiş sayılır."
        )
        reply = "İş Kanunu Madde 17'ye göre ihbar süresi iki haftadır."
        print("Grounding skoru:", await output_guardrail(context, reply))

    asyncio.run(_test())

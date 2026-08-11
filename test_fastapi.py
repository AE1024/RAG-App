from fastapi import FastAPI
from openai import OpenAI

app = FastAPI(title="LM Studio FastAPI Integration", description="FastAPI communicating with local LM Studio model")

# LM Studio yerel sunucu adresi ve sahte API anahtarı (LM Studio key sormaz ama kütüphane ister)
client = OpenAI(
    base_url="http://192.168.56.1:1234/v1",
    api_key="lm-studio",
    max_retries=3
)

@app.post("/generate")
def generate_text(prompt: str):
    try:
        # LM Studio'da şu an aktif olan modeli otomatik algılar veya model adını yazabilirsin
        response = client.chat.completions.create(
            model="turkish-gemma-9b-t1:2",  # LM Studio genellikle bu genel adı kullanır
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return {
            "status": "success",
            "response": response.choices[0].message.content
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def root():
    return {"message": "Test!"}
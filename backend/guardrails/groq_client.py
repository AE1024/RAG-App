import os
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

class Client_Groq:
    def __init__(self, model_name="openai/gpt-oss-120b"):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.async_client = AsyncGroq(api_key=self.groq_api_key)
        self._model_name = model_name

    async def get_chat_response_async(self, messages, temperature=0):
        response = await self.async_client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
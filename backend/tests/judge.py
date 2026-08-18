"""
DeepEval judge modeli — Groq (openai/gpt-oss-120b).

DeepEval özel model için DeepEvalBaseLLM zorunlu; bu sınıf
sırf o köprüyü kurar. Schema geldiğinde JSON mode açılır,
gelmeyince normal text döner.
"""
import json
import os
import re

from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq


class GroqJudge(DeepEvalBaseLLM):
    _MODEL = "openai/gpt-oss-120b"

    def __init__(self):
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema=None):
        kwargs = {
            "model": self._MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        content = self._client.chat.completions.create(**kwargs).choices[0].message.content

        if schema is not None:
            try:
                return schema(**json.loads(content))
            except Exception:
                m = re.search(r"\{.*\}", content, re.DOTALL)
                if m:
                    return schema(**json.loads(m.group()))
                raise

        return content

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self._MODEL


judge = GroqJudge()

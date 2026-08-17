"""
LangSmith tracing — tek giriş noktası.

Kullanım:
    from backend.tracing import traceable, wrap_openai

    # OpenAI client'ı otomatik izlemek için:
    client = wrap_openai(OpenAI(...))

    # Herhangi bir fonksiyonu izlemek için:
    @traceable(name="retrieval", run_type="retriever")
    def search(...): ...

LANGSMITH_TRACING env değişkeni set edilmemişse
langsmith otomatik no-op çalışır, ek kontrol gerekmez.
"""
from langsmith import traceable
from langsmith.wrappers import wrap_openai

__all__ = ["traceable", "wrap_openai"]

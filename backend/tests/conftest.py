"""
Test ortamı kurulumu — backend/tests/ altındaki tüm testler için otomatik yüklenir.

backend.main modül seviyesinde Retriever(), OpenAI() ve ForwardGuardPipeline()
başlatır. Bu dosya bu ağır nesneleri mock'layarak saf fonksiyonların
(clean_reply, build_context vb.) import edilebilmesini sağlar.
"""
import sys
from unittest.mock import MagicMock

# 1. backend.tracing önce hazır olmalı — RAG.retrieval onu import eder
_tracing = MagicMock()
_tracing.traceable = lambda **kw: (lambda f: f)
_tracing.wrap_openai = lambda x: x
sys.modules.setdefault("backend.tracing", _tracing)

# 2. RAG.retrieval gerçek modül olarak yüklenir; yalnızca Retriever sınıfı mock'lanır
import RAG.retrieval as _rag_retrieval  # noqa: E402
_rag_retrieval.Retriever = MagicMock()

# 3. Kalan ağır bağımlılıklar
for _mod in [
    "backend.guardrails.forward_guards",
    "backend.guardrails.input_guards",
    "backend.guardrails.output_guards",
    "backend.logger",
    "openai",
]:
    sys.modules.setdefault(_mod, MagicMock())

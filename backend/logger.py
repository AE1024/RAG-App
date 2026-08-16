"""
logger.py — Structured JSONL logging for the RAG chat pipeline.

Her /chat isteği için tek satır JSON yazar: logs/chat.jsonl
PII içeren sorgular PIIGuard.mask() ile maskelenerek kaydedilir.
"""
import json
import logging
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.guardrails.forward_guards import PIIGuard

_log_dir = Path(__file__).resolve().parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)

_logger = logging.getLogger("rag.chat")
_logger.setLevel(logging.INFO)
_logger.propagate = False

_file_handler = logging.FileHandler(_log_dir / "chat.jsonl", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(message)s"))
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(message)s"))
_logger.addHandler(_file_handler)
_logger.addHandler(_console_handler)


class RequestLogger:
    """
    Tek bir /chat isteğinin log kaydını tutar.
    Kullanım:
        rl = RequestLogger(request.message, request.session_id)
        rl.set(guard_pre="block:...")
        rl.flush()          ← isteğin sonunda bir kez çağrılır
    """

    def __init__(self, message: str, session_id: str) -> None:
        self._t0 = time.monotonic()
        self._record: dict = {
            "ts":                datetime.now(timezone.utc).isoformat(),
            "request_id":        str(uuid.uuid4()),
            "session_id":        session_id,
            "query_masked":      PIIGuard.mask(message),
            "guard_pre":         "pass",
            "guard_empty":       "pass",
            "guard_post":        "pass",
            "retrieval_count":   0,
            "sources_count":     0,
            "tokens_prompt":     None,
            "tokens_completion": None,
            "latency_ms":        None,
            "early_exit":        None,
        }

    def set(self, **kwargs) -> None:
        self._record.update(kwargs)

    def flush(self) -> None:
        self._record["latency_ms"] = round((time.monotonic() - self._t0) * 1000)
        _logger.info(json.dumps(self._record, ensure_ascii=False))

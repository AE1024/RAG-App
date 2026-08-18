"""
Unit testler — saf yardımcı fonksiyonlar (LLM / API / ChromaDB gerektirmez).
Çalıştırma: pytest backend/tests/unit_tests/ -v
"""
from langchain_core.documents import Document

from backend.main import Source, build_context, clean_reply, filter_grounded_sources
from RAG.retrieval import _normalize_query

# ── _normalize_query ─────────────────────────────────────────────────────────


def test_normalize_zaman_asimi_with_space():
    assert _normalize_query("zaman aşımı davası") == "zamanaşımı davası"


def test_normalize_zaman_asimi_uppercase():
    assert _normalize_query("ZAMAN AŞIMI süresi") == "zamanaşımı süresi"


def test_normalize_no_change_needed():
    assert _normalize_query("ihbar süresi nedir") == "ihbar süresi nedir"


def test_normalize_haksiz_fiil_preserved():
    result = _normalize_query("haksız fiil tazminatı")
    assert "haksız fiil" in result


def test_normalize_multiple_patterns():
    query = "zaman aşımı ile sebepsiz zenginleşme ilişkisi"
    result = _normalize_query(query)
    assert "zamanaşımı" in result
    assert "sebepsiz zenginleşme" in result


# ── clean_reply ──────────────────────────────────────────────────────────────


def test_clean_reply_removes_think_block():
    assert clean_reply("<think>İç düşünce.</think>Asıl cevap.") == "Asıl cevap."


def test_clean_reply_removes_multiline_think():
    raw = "<think>\nSatır 1\nSatır 2\n</think>\n\nSon cevap."
    assert clean_reply(raw) == "Son cevap."


def test_clean_reply_no_think_block():
    assert clean_reply("Normal bir yanıt metni.") == "Normal bir yanıt metni."


def test_clean_reply_strips_whitespace():
    assert clean_reply("  <think>x</think>  Yanıt.  ") == "Yanıt."


def test_clean_reply_removes_artifact_pattern():
    raw = "[] : artifact\nGerçek cevap."
    result = clean_reply(raw)
    assert "[] :" not in result
    assert "Gerçek cevap." in result


# ── filter_grounded_sources ──────────────────────────────────────────────────


def _src(law_name: str) -> Source:
    return Source(law_name=law_name, law=law_name, source="test.pdf")


def test_filter_returns_empty_on_not_found():
    result = filter_grounded_sources(
        "Elimdeki kanun metinlerinde bilgi bulunamadı.",
        [_src("İş Kanunu (4857) Madde 17")],
    )
    assert result == []


def test_filter_returns_empty_on_not_found_variant():
    result = filter_grounded_sources(
        "Bu konuya dair bilgi yer almamaktadır.",
        [_src("Nüfus Hizmetleri Kanunu (5490) Madde 15")],
    )
    assert result == []


def test_filter_matches_madde_number_in_reply():
    
    s17 = Source(law_name="İş Kanunu", law="İş Kanunu Madde 17", source="test.pdf")
    s48 = Source(law_name="Tüketici Kanunu", law="Tüketici Kanunu Madde 48", source="test.pdf")
    result = filter_grounded_sources(
        "İş Kanunu Madde 17 uyarınca ihbar süresi iki haftadır.",
        [s17, s48],
    )
    assert s17 in result
    assert s48 not in result


def test_filter_fallback_returns_all_when_no_madde_in_reply():
    sources = [_src("Kanun A Madde 1"), _src("Kanun B Madde 2")]
    result = filter_grounded_sources("Genel bir açıklama yapılabilir.", sources)
    assert result == sources


# ── build_context ─────────────────────────────────────────────────────────────


def _doc(kanun_adi: str, madde: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "kanun_adi": kanun_adi,
            "kanun_tam": f"{kanun_adi} (test)",
            "madde": madde,
            "kaynak": f"{kanun_adi}.pdf",
        },
    )


def test_build_context_two_different_articles():
    docs = [
        _doc("iskanunu", "Madde 17", "İhbar süresi iki haftadır."),
        _doc("iskanunu", "Madde 18", "Fesih geçerli sebebe dayanmalıdır."),
    ]
    ctx, sources = build_context(docs)
    assert "Madde 17" in ctx
    assert "Madde 18" in ctx
    assert len(sources) == 2


def test_build_context_deduplicates_identical_chunks():
    doc = _doc("iskanunu", "Madde 17", "İhbar süresi iki haftadır.")
    ctx, sources = build_context([doc, doc])
    assert ctx.count("Madde 17") == 1
    assert len(sources) == 1


def test_build_context_truncates_long_text():
    long_text = "A" * 4000  # MAX_PARENT_CHARS (3000) üstü
    ctx, _ = build_context([_doc("iskanunu", "Madde 1", long_text)])
    assert "[...]" in ctx


def test_build_context_empty_list():
    ctx, sources = build_context([])
    assert ctx == ""
    assert sources == []

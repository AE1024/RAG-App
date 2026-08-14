from langchain_huggingface import HuggingFaceEmbeddings

EMBED_MODEL = "intfloat/multilingual-e5-small"  # 120MB, 384 dim, hızlı


class E5SmallEmbeddings(HuggingFaceEmbeddings):
    """multilingual-e5-small için passage:/query: prefix yönetimi."""

    def __init__(self, **kwargs):
        kwargs.setdefault("model_name", EMBED_MODEL)
        kwargs.setdefault("encode_kwargs", {"normalize_embeddings": True})
        super().__init__(**kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents(["passage: " + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query("query: " + text)

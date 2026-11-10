import logging
import threading
import time
from pathlib import Path
from uuid import uuid4

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone , ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings
from app.rag.hybrid import Bm25Index, HybridRetriever

logger = logging.getLogger(__name__)
settings = get_settings()


_embeddings = None
_vectorstore = None
_bm25_index = None
_bm25_mtime = None
_bm25_lock = threading.Lock()


EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
}



def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()
    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()
    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]
    if "text-embedding-3-small" in normalized:
        return 1536
    if "text-embedding-3-large" in normalized:
        return 3072
    if "text-embedding-ada-002" in normalized:
        return 1536
    if "all-minilm" in normalized:
        return 384

    raise ValueError(
        f"Unsupported embedding model '{model_name or settings.embedding_model}' for Pinecone. "
        "Add the matching dimension to EMBEDDING_DIMENSIONS."
    )



def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # FastEmbed runs the same MiniLM weights through ONNX, which keeps the
        # deployment inside the memory limit of a small instance.
        _embeddings = FastEmbedEmbeddings(model_name=settings.embedding_model)
    return _embeddings




def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    names = {x["name"] for x in pc.list_indexes()}

    if settings.pinecone_index_name in names:
        index_info = pc.describe_index(settings.pinecone_index_name)
        current_dimension = getattr(index_info, "dimension", None)
        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        if current_dimension is not None and current_dimension != desired_dimension:
            # Never delete a live index from a lazy initializer; that destroys the corpus.
            raise RuntimeError(
                f"Pinecone index '{settings.pinecone_index_name}' has dimension "
                f"{current_dimension}, but embedding model '{settings.embedding_model}' "
                f"produces {desired_dimension}. Point PINECONE_INDEX_NAME at a matching "
                "index (or re-ingest into a new one) instead of changing this process."
            )
        return pc.Index(settings.pinecone_index_name)

    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=desired_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    deadline = time.monotonic() + 120
    while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"Pinecone index '{settings.pinecone_index_name}' was not ready within 120s"
            )
        time.sleep(1)

    return pc.Index(settings.pinecone_index_name)





def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
    return _vectorstore



def _file_mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def get_bm25_index():
    global _bm25_index, _bm25_mtime
    path = Path(settings.bm25_index_path)
    mtime = _file_mtime(path)
    with _bm25_lock:
        # Reload when ingestion in another worker rewrites the shared corpus file.
        if _bm25_index is None or _bm25_mtime != mtime:
            _bm25_index = Bm25Index(path)
            _bm25_mtime = mtime
            if len(_bm25_index) == 0:
                logger.warning(
                    "BM25 index %s is empty; retrieval is vector-only until documents are ingested",
                    settings.bm25_index_path,
                )
        return _bm25_index




def get_retriever():
    return HybridRetriever(
        get_vectorstore().as_retriever(search_kwargs={"k": max(settings.top_k * 2, 10)}),
        get_bm25_index(),
        settings.top_k,
    )



def add_documents(chunks):
    ids = get_vectorstore().add_documents(chunks)
    try:
        get_bm25_index().add(chunks)
    except Exception:
        logger.warning("Failed to update BM25 corpus; keyword retrieval is stale", exc_info=True)
    return ids


def replace_documents(chunks: list[Document]) -> list[str]:
    sources = {str(chunk.metadata.get("source", "")) for chunk in chunks}
    if not chunks or len(sources) != 1 or not next(iter(sources)):
        raise ValueError("Replacement chunks must share one non-empty source")

    source = next(iter(sources))
    ingestion_id = uuid4().hex
    versioned_chunks = [
        Document(
            page_content=chunk.page_content,
            metadata={**chunk.metadata, "ingestion_id": ingestion_id},
        )
        for chunk in chunks
    ]

    store = get_vectorstore()
    ids = store.add_documents(versioned_chunks)
    try:
        store.delete(
            filter={
                "$and": [
                    {"source": {"$eq": source}},
                    {
                        "$or": [
                            {"ingestion_id": {"$exists": False}},
                            {"ingestion_id": {"$ne": ingestion_id}},
                        ]
                    },
                ]
            }
        )
    except Exception:
        logger.exception("Failed to remove previous vectors for %s", source)
        if ids:
            try:
                store.delete(ids=ids)
            except Exception:
                logger.exception("Failed to roll back new vectors for %s", source)
        raise

    try:
        get_bm25_index().replace_source(versioned_chunks)
    except Exception:
        logger.warning(
            "Failed to replace BM25 source %s; keyword retrieval is stale",
            source,
            exc_info=True,
        )
    return ids

import hashlib
import json
import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25L

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
RRF_K = 40


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def chunk_key(document: Document) -> str:
    """Stable identity shared by the vector and keyword indexes."""
    source = str(document.metadata.get("source", ""))
    start_index = document.metadata.get("start_index")
    if start_index is not None:
        return f"{source}#{start_index}"
    digest = hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()
    return f"{source}#{digest}"


def reciprocal_rank_fusion(rankings: list[list[Document]], rrf_k: int = RRF_K) -> list[Document]:
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranking in rankings:
        for rank, document in enumerate(ranking, start=1):
            key = chunk_key(document)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            documents.setdefault(key, document)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [documents[key] for key in ordered]


class Bm25Index:
    """Persisted BM25 corpus over the same chunks stored in Pinecone."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records = self._load()
        self._bm25: BM25L | None = None
        self._build()

    def __len__(self) -> int:
        return len(self._records)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid BM25 corpus at {self.path}:{line_number}") from exc
        return records

    def _build(self) -> None:
        corpus = [tokenize(record["text"]) for record in self._records]
        # BM25L keeps a non-zero IDF on the small corpora where BM25Okapi scores everything zero.
        self._bm25 = BM25L(corpus) if any(corpus) else None

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in self._records),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def add(self, documents: list[Document]) -> None:
        by_key = {record["id"]: record for record in self._records}
        for document in documents:
            key = chunk_key(document)
            by_key[key] = {
                "id": key,
                "text": document.page_content,
                "metadata": document.metadata,
            }
        self._records = list(by_key.values())
        self._write()
        self._build()

    def replace_source(self, documents: list[Document]) -> None:
        sources = {str(document.metadata.get("source", "")) for document in documents}
        if len(sources) != 1 or not next(iter(sources)):
            raise ValueError("Replacement documents must share one non-empty source")

        source = next(iter(sources))
        by_key = {
            record["id"]: record
            for record in self._records
            if record.get("metadata", {}).get("source") != source
        }
        for document in documents:
            key = chunk_key(document)
            by_key[key] = {
                "id": key,
                "text": document.page_content,
                "metadata": document.metadata,
            }

        previous_records = self._records
        self._records = list(by_key.values())
        try:
            self._write()
        except Exception:
            self._records = previous_records
            raise
        self._build()

    def search(self, query: str, k: int) -> list[Document]:
        if self._bm25 is None:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        results = []
        for index in ranked[:k]:
            if scores[index] <= 0:
                break
            record = self._records[index]
            results.append(Document(page_content=record["text"], metadata=record["metadata"]))
        return results


class HybridRetriever:
    def __init__(self, vector_retriever, bm25_index: Bm25Index, top_k: int = 5):
        self.vector_retriever = vector_retriever
        self.bm25_index = bm25_index
        self.top_k = top_k
        self.candidate_k = max(top_k * 2, 10)

    def invoke(self, query: str) -> list[Document]:
        vector_documents = self.vector_retriever.invoke(query)
        if len(self.bm25_index) == 0:
            logger.warning("BM25 index is empty; using vector-only retrieval")
            return list(vector_documents)[: self.top_k]
        bm25_documents = self.bm25_index.search(query, self.candidate_k)
        fused = reciprocal_rank_fusion([vector_documents, bm25_documents])
        return fused[: self.top_k]

from pathlib import Path

from langchain_core.documents import Document

from app.rag.hybrid import (
    Bm25Index,
    HybridRetriever,
    chunk_key,
    reciprocal_rank_fusion,
    tokenize,
)


def make_document(text: str, start_index: int = 0, source: str = "policy.md") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "start_index": start_index},
    )


class FakeVectorRetriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.queries = []

    def invoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return self.documents


def test_tokenize_lowercases_and_splits_words():
    assert tokenize("VPN access: MFA-required!") == ["vpn", "access", "mfa", "required"]


def test_chunk_key_uses_source_and_start_index():
    first = chunk_key(make_document("first", start_index=0))
    second = chunk_key(make_document("second", start_index=900))

    assert first == "policy.md#0"
    assert second == "policy.md#900"
    assert first != second


def test_reciprocal_rank_fusion_rewards_documents_in_both_lists():
    first = make_document("first", start_index=0)
    shared = make_document("shared", start_index=900)
    second = make_document("second", start_index=1800)

    fused = reciprocal_rank_fusion([[first, shared], [shared, second]])

    assert fused == [shared, first, second]


def test_bm25_index_persists_and_ranks_matching_chunks(tmp_path: Path):
    path = tmp_path / "bm25_corpus.jsonl"
    index = Bm25Index(path)
    index.add(
        [
            make_document("Reset your VPN password in the portal.", start_index=0),
            make_document("Replace printer toner every quarter.", start_index=900),
        ]
    )

    reloaded = Bm25Index(path)
    results = reloaded.search("vpn password reset", k=2)

    assert len(reloaded) == 2
    assert results[0].page_content.startswith("Reset your VPN password")


def test_bm25_index_replaces_reingested_chunk(tmp_path: Path):
    index = Bm25Index(tmp_path / "bm25_corpus.jsonl")
    index.add([make_document("old content", start_index=0)])
    index.add([make_document("new content", start_index=0)])

    assert len(index) == 1
    assert index.search("new", k=1)[0].page_content == "new content"


def test_bm25_index_replaces_all_chunks_for_source(tmp_path: Path):
    index = Bm25Index(tmp_path / "bm25_corpus.jsonl")
    index.add(
        [
            make_document("old first chunk", start_index=0),
            make_document("old second chunk", start_index=900),
            make_document("keep this other document", source="other.md", start_index=0),
        ]
    )
    index.replace_source([make_document("new first chunk", start_index=0)])

    assert len(index) == 2
    assert index.search("second", k=2) == []
    assert index.search("new", k=1)[0].page_content == "new first chunk"
    assert index.search("keep", k=1)[0].page_content == "keep this other document"


def test_hybrid_retriever_fuses_and_caps_results(tmp_path: Path):
    index = Bm25Index(tmp_path / "bm25_corpus.jsonl")
    index.add(
        [
            make_document("VPN password reset", start_index=0),
            make_document("Printer toner replacement", start_index=900),
        ]
    )
    vector = FakeVectorRetriever(
        [
            make_document("Printer toner replacement", start_index=900),
            make_document("VPN password reset", start_index=0),
        ]
    )

    results = HybridRetriever(vector, index, top_k=2).invoke("vpn password reset")

    assert vector.queries == ["vpn password reset"]
    assert len(results) == 2
    assert results[0].page_content == "VPN password reset"


def test_hybrid_retriever_dedupes_identical_chunks(tmp_path: Path):
    index = Bm25Index(tmp_path / "bm25_corpus.jsonl")
    index.add(
        [
            make_document("VPN password reset", start_index=0, source="a.md"),
            make_document("Printer toner replacement", start_index=900, source="a.md"),
        ]
    )
    vector = FakeVectorRetriever(
        [
            make_document("VPN password reset", start_index=0, source="a.md"),
            make_document("VPN password reset", start_index=0, source="b.md"),
            make_document("Printer toner replacement", start_index=900, source="a.md"),
        ]
    )

    results = HybridRetriever(vector, index, top_k=3).invoke("vpn password reset")

    contents = [document.page_content for document in results]
    assert contents.count("VPN password reset") == 1
    assert len(results) == 2


def test_hybrid_retriever_degrades_to_vector_when_bm25_is_empty(tmp_path: Path):
    vector_documents = [make_document("first"), make_document("second")]
    retriever = HybridRetriever(
        FakeVectorRetriever(vector_documents),
        Bm25Index(tmp_path / "missing.jsonl"),
        top_k=1,
    )

    assert retriever.invoke("anything") == vector_documents[:1]


def test_add_documents_updates_both_indexes(monkeypatch, tmp_path: Path):
    from app.rag import vectorstore

    added = []

    class FakeVectorStore:
        def add_documents(self, chunks):
            return ["vector-id"]

    class FakeBm25Index:
        def add(self, chunks):
            added.extend(chunks)

    monkeypatch.setattr(vectorstore, "get_vectorstore", lambda: FakeVectorStore())
    monkeypatch.setattr(vectorstore, "get_bm25_index", lambda: FakeBm25Index())
    chunks = [make_document("VPN access requires MFA.")]

    assert vectorstore.add_documents(chunks) == ["vector-id"]
    assert added == chunks


def test_add_documents_keeps_vector_ingestion_when_bm25_update_fails(monkeypatch):
    from app.rag import vectorstore

    class FakeVectorStore:
        def add_documents(self, chunks):
            return ["vector-id"]

    class BrokenBm25Index:
        def add(self, chunks):
            raise RuntimeError("disk full")

    monkeypatch.setattr(vectorstore, "get_vectorstore", lambda: FakeVectorStore())
    monkeypatch.setattr(vectorstore, "get_bm25_index", lambda: BrokenBm25Index())

    assert vectorstore.add_documents([make_document("x")]) == ["vector-id"]


def test_replace_documents_upserts_new_chunks_and_removes_previous_source(monkeypatch):
    from app.rag import vectorstore

    class FakeVectorStore:
        def __init__(self):
            self.added = []
            self.deleted = []

        def add_documents(self, chunks):
            self.added.extend(chunks)
            return ["new-vector-id"]

        def delete(self, **kwargs):
            self.deleted.append(kwargs)

    class FakeBm25Index:
        def __init__(self):
            self.replaced = []

        def replace_source(self, chunks):
            self.replaced.extend(chunks)

    fake_store = FakeVectorStore()
    fake_bm25 = FakeBm25Index()
    monkeypatch.setattr(vectorstore, "get_vectorstore", lambda: fake_store)
    monkeypatch.setattr(vectorstore, "get_bm25_index", lambda: fake_bm25)
    chunks = [make_document("Updated VPN guidance", start_index=0)]

    assert vectorstore.replace_documents(chunks) == ["new-vector-id"]
    assert fake_store.added[0].page_content == "Updated VPN guidance"
    ingestion_id = fake_store.added[0].metadata["ingestion_id"]
    assert fake_store.deleted == [
        {
            "filter": {
                "$and": [
                    {"source": {"$eq": chunks[0].metadata["source"]}},
                    {
                        "$or": [
                            {"ingestion_id": {"$exists": False}},
                            {"ingestion_id": {"$ne": ingestion_id}},
                        ]
                    },
                ]
            }
        }
    ]
    assert fake_bm25.replaced == fake_store.added

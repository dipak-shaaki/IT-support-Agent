import json
import os

import pytest

from app.rag import vectorstore


def test_ensure_index_refuses_dimension_mismatch_without_deleting(monkeypatch):
    class FakeIndexInfo:
        dimension = 999

    class FakePinecone:
        deleted = False

        def list_indexes(self):
            return [{"name": vectorstore.settings.pinecone_index_name}]

        def describe_index(self, name):
            return FakeIndexInfo()

        def delete_index(self, name):
            self.deleted = True

    fake = FakePinecone()
    monkeypatch.setattr(vectorstore.settings, "pinecone_api_key", "test-key")
    monkeypatch.setattr(vectorstore, "Pinecone", lambda api_key: fake)

    with pytest.raises(RuntimeError, match="has dimension 999"):
        vectorstore.ensure_index()

    assert fake.deleted is False


def test_bm25_cache_reloads_when_corpus_file_changes(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    record = {"id": "vpn.md#0", "text": "connect to vpn", "metadata": {"source": "vpn.md"}}
    corpus.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(vectorstore.settings, "bm25_index_path", str(corpus))
    monkeypatch.setattr(vectorstore, "_bm25_index", None)
    monkeypatch.setattr(vectorstore, "_bm25_mtime", None)

    first = vectorstore.get_bm25_index()
    second_record = {
        "id": "mfa.md#0",
        "text": "reset mfa",
        "metadata": {"source": "mfa.md"},
    }
    corpus.write_text(
        json.dumps(record) + "\n" + json.dumps(second_record) + "\n", encoding="utf-8"
    )
    mtime = corpus.stat().st_mtime_ns
    os.utime(corpus, ns=(mtime + 1_000_000, mtime + 1_000_000))

    second = vectorstore.get_bm25_index()

    assert second is not first
    assert len(second) == 2

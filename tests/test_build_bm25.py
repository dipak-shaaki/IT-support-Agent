from pathlib import Path

import pytest

from app.build_bm25 import build
from app.rag.hybrid import Bm25Index


def test_build_writes_a_searchable_corpus_without_embeddings(tmp_path: Path):
    source = tmp_path / "kb"
    source.mkdir()
    (source / "records.md").write_text(
        "## IT-00001\n"
        "- id: IT-00001\n"
        "- record_number: 1\n"
        "### Question\n"
        "How do I connect to the VPN?\n"
        "### Answer\n"
        "Install the NovaSecure client.\n",
        encoding="utf-8",
    )
    (source / "runbook.txt").write_text(
        "Reset a password from the service desk portal.", encoding="utf-8"
    )
    output = tmp_path / "bm25.jsonl"

    count = build(source, output)

    assert count == 2
    results = Bm25Index(output).search("NovaSecure", 1)
    assert results[0].metadata["record_id"] == "IT-00001"


def test_bm25_index_rejects_symlinked_corpus(tmp_path: Path):
    target = tmp_path / "outside.jsonl"
    target.write_text('{"id":"safe","text":"safe","metadata":{}}\n')
    link = tmp_path / "corpus.jsonl"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        Bm25Index(link)


def test_build_rejects_symlink_output_without_touching_target(tmp_path: Path):
    source = tmp_path / "kb"
    source.mkdir()
    (source / "note.md").write_text("hello", encoding="utf-8")
    target = tmp_path / "outside.jsonl"
    target.write_text("keep me", encoding="utf-8")
    link = tmp_path / "corpus.jsonl"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        build(source, link)

    assert target.read_text(encoding="utf-8") == "keep me"

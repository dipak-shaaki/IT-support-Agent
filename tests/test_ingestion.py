from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.services import ingestion
from app.services.ingestion import SUPPORTED, chunk_documents, load_file


def test_loads_text_file(tmp_path: Path):
    source = tmp_path / "policy.txt"
    source.write_text("Use the company VPN client.", encoding="utf-8")

    documents = load_file(source)

    assert len(documents) == 1
    assert documents[0].page_content == "Use the company VPN client."
    assert documents[0].metadata["source"] == str(source)


def test_loads_docx_paragraphs(tmp_path: Path):
    from docx import Document as DocxDocument

    source = tmp_path / "policy.docx"
    document = DocxDocument()
    document.add_paragraph("VPN access requires MFA.")
    document.add_paragraph("")
    document.add_paragraph("Contact the service desk for help.")
    document.save(source)

    documents = load_file(source)

    assert len(documents) == 1
    assert documents[0].page_content == (
        "VPN access requires MFA.\nContact the service desk for help."
    )


def test_load_file_rejects_unsupported_extension(tmp_path: Path):
    source = tmp_path / "policy.csv"
    source.write_text("policy", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type: .csv"):
        load_file(source)


def test_supported_extensions_include_document_formats():
    assert SUPPORTED == {".pdf", ".txt", ".md", ".docx"}


def test_chunk_documents_preserves_source_metadata():
    source = Document(page_content="VPN " * 500, metadata={"source": "vpn.md"})

    chunks = chunk_documents([source])

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "vpn.md" for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


def test_chunk_documents_keeps_a_runbook_section_together():
    source = Document(
        page_content=(
            "## Email Troubleshooting\n"
            + "email guidance " * 60
            + "confirm whether the issue affects one user or many users. "
            + "whether sending, receiving, or both are affected.\n\n"
            "## Laptop Performance\n"
            + "laptop guidance " * 60
        ),
        metadata={"source": "runbook.md"},
    )

    chunks = chunk_documents([source])
    combined = [chunk.page_content for chunk in chunks]

    assert any(
        "one user or many users" in text and "sending, receiving, or both" in text
        for text in combined
    )


def test_loads_markdown_qa_records_with_metadata(tmp_path: Path):
    source = tmp_path / "it_support_01.md"
    source.write_text(
        "---\n"
        'source: "IT Support Dataset"\n'
        'category: "IT Support"\n'
        "---\n\n"
        "# IT Support Knowledge Base\n\n"
        "## IT-00001\n\n"
        "### Metadata\n\n"
        "- id: IT-00001\n"
        "- source: IT Support Dataset\n"
        "- category: IT Support\n"
        "- file: it_support_01.md\n"
        "- record_number: 1\n"
        "- document_type: question_answer\n"
        "- language: en\n\n"
        "### Question\n\nWhat is a VPN?\n\n"
        "### Answer\n\nA private company network.\n\n---\n\n"
        "## IT-00002\n\n"
        "### Metadata\n\n"
        "- id: IT-00002\n"
        "- record_number: 2\n\n"
        "### Question\n\nWhat is MFA?\n\n"
        "### Answer\n\nA second sign-in factor.\n\n---\n",
        encoding="utf-8",
    )

    documents = load_file(source)

    assert len(documents) == 2
    first = documents[0]
    assert first.metadata["record_id"] == "IT-00001"
    assert first.metadata["source"] == str(source)
    assert first.metadata["category"] == "IT Support"
    assert first.metadata["record_number"] == 1
    assert first.metadata["question"] == "What is a VPN?"
    assert first.page_content == "What is a VPN?\n\nA private company network."

    chunks = chunk_documents(documents)

    assert len(chunks) == 2
    assert {chunk.metadata["record_id"] for chunk in chunks} == {
        "IT-00001",
        "IT-00002",
    }
    assert chunks[0].metadata["start_index"] != chunks[1].metadata["start_index"]


def test_plain_markdown_still_uses_the_text_loader(tmp_path: Path):
    source = tmp_path / "policy.md"
    source.write_text("# VPN\n\nUse the VPN client.", encoding="utf-8")

    documents = load_file(source)

    assert len(documents) == 1
    assert documents[0].metadata["source"] == str(source)
    assert "Use the VPN client." in documents[0].page_content


def test_index_file_preserves_the_sanitized_upload_name(monkeypatch, tmp_path: Path):
    source = tmp_path / "random-upload-id.md"
    document = Document(page_content="VPN steps", metadata={"source": str(source)})
    captured = []
    monkeypatch.setattr(ingestion, "get_settings", lambda: type("Settings", (), {"upload_dir": str(tmp_path)})())
    monkeypatch.setattr(ingestion, "load_file", lambda path: [document])
    monkeypatch.setattr(
        ingestion, "replace_documents", lambda chunks: captured.extend(chunks) or ["vector-1"]
    )

    count, ids = ingestion.index_file(source, "vpn.md")

    assert count == 1
    assert ids == ["vector-1"]
    assert captured[0].metadata["original_filename"] == "vpn.md"
    assert captured[0].metadata["file"] == "vpn.md"
    assert captured[0].metadata["source"] == str(tmp_path / "vpn.md")

from pathlib import Path

import pytest
from langchain_core.documents import Document

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

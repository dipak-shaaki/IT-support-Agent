from pathlib import Path
from typing import Iterable

from docx import Document as DocxDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.rag.vectorstore import replace_documents

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 150
# Records in the IT Support dataset are at most ~2,300 characters, so one Q&A
# fits in a single chunk; larger records are split and the question repeated.
RECORD_CHUNK_SIZE = 2500
RECORD_CHUNK_OVERLAP = 200

class EmptyDocumentError(ValueError):
    """Raised when an upload has nothing that can be indexed."""


def load_markdown_records(path: Path) -> list[Document]:
    """Parse the IT Support Q&A Markdown format into one Document per record.

    The source file keeps the file path in ``source`` so replacement and the
    BM25 index still group all records from the same document.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    line_offsets = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line) + 1
    headings = [
        (line_number, record_id)
        for line_number, line in enumerate(lines)
        if (record_id := _record_id(line)) is not None
    ]
    if not headings:
        return []

    documents = []
    for index, (start, record_id) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
        document = _markdown_record(lines[start:end], record_id, path, line_offsets[start])
        if document is not None:
            documents.append(document)
    return documents


def _markdown_record(block: list[str], record_id: str, path: Path, start_index: int) -> Document | None:
    question_header = _section_line(block, "### Question")
    if question_header is None:
        return None
    answer_header = _section_line(block, "### Answer", question_header + 1)
    if answer_header is None:
        return None
    question = "\n".join(block[question_header + 1:answer_header]).strip()
    answer_lines = block[answer_header + 1:]
    while answer_lines and not answer_lines[-1].strip():
        answer_lines.pop()
    if answer_lines and len(answer_lines[-1].strip()) >= 3 and set(answer_lines[-1].strip()) == {"-"}:
        answer_lines.pop()
    answer = "\n".join(answer_lines).strip()
    if not question or not answer:
        return None

    meta = _metadata_from_lines(block)
    metadata = {
        "source": str(path),
        "record_id": meta.get("id") or record_id,
        "question": question,
        "dataset_source": meta.get("source", ""),
        "category": meta.get("category", ""),
        "file": meta.get("file", path.name),
        "document_type": meta.get("document_type", ""),
        "language": meta.get("language", ""),
        "start_index": start_index,
    }
    record_number = meta.get("record_number", "")
    if record_number.isdigit():
        metadata["record_number"] = int(record_number)
    return Document(page_content=f"{question}\n\n{answer}", metadata=metadata)


def _record_id(line: str) -> str | None:
    if not line.startswith("## IT-"):
        return None
    suffix = line[6:].rstrip(" \t")
    if suffix and all(character.isdigit() for character in suffix):
        return f"IT-{suffix}"
    return None


def _section_line(lines: list[str], marker: str, start: int = 0) -> int | None:
    return next(
        (index for index in range(start, len(lines)) if lines[index].strip() == marker),
        None,
    )


def _metadata_from_lines(lines: list[str]) -> dict[str, str]:
    metadata = {}
    for line in lines:
        if not line.startswith("- "):
            continue
        key, separator, value = line[2:].partition(":")
        if (
            separator
            and key
            and all(character.isascii() and (character.isalpha() or character == "_") for character in key)
        ):
            metadata[key] = value.strip(" \t")
    return metadata


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".txt", ".md"}:
        if suffix == ".md":
            records = load_markdown_records(path)
            if records:
                return records
        return TextLoader(str(path), encoding="utf-8").load()
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [Document(page_content=text, metadata={"source": str(path)})]
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_documents(docs: Iterable[Document]) -> list[Document]:
    documents = list(docs)
    record_documents = [
        document for document in documents if document.metadata.get("record_id")
    ]
    regular_documents = [
        document for document in documents if not document.metadata.get("record_id")
    ]
    chunks: list[Document] = []

    if regular_documents:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )
        chunks.extend(splitter.split_documents(regular_documents))

    if record_documents:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RECORD_CHUNK_SIZE,
            chunk_overlap=RECORD_CHUNK_OVERLAP,
            add_start_index=True,
        )
        for document in record_documents:
            base_start = int(document.metadata.get("start_index", 0) or 0)
            question = document.metadata.get("question", "")
            for piece_index, piece in enumerate(splitter.split_documents([document])):
                # Keep chunk keys unique across records in the same file.
                piece.metadata["start_index"] = base_start + int(
                    piece.metadata.get("start_index", 0) or 0
                )
                if piece_index and question:
                    piece.page_content = f"{question}\n\n{piece.page_content}"
                chunks.append(piece)

    return chunks


def index_file(
    path: Path, original_filename: str | None = None
) -> tuple[int, list[str]]:
    """Parse, chunk, embed and store one document. Used by inline and queued ingestion."""
    documents = load_file(path)
    if original_filename:
        source = str(Path(get_settings().upload_dir).resolve() / original_filename)
        for document in documents:
            document.metadata["source"] = source
            document.metadata["file"] = original_filename
            document.metadata["original_filename"] = original_filename
    chunks = chunk_documents(documents)
    if not chunks:
        raise EmptyDocumentError("The uploaded document contains no indexable text")
    return len(chunks), replace_documents(chunks)

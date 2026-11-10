import re
from pathlib import Path
from typing import Iterable

from docx import Document as DocxDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.vectorstore import replace_documents

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 150
# Records in the IT Support dataset are at most ~2,300 characters, so one Q&A
# fits in a single chunk; larger records are split and the question repeated.
RECORD_CHUNK_SIZE = 2500
RECORD_CHUNK_OVERLAP = 200

RECORD_HEADING = re.compile(r"^## (IT-\d+)\s*$", re.M)
METADATA_LINE = re.compile(r"^- ([A-Za-z_]+):\s*(.*?)\s*$", re.M)
QUESTION_SECTION = re.compile(r"### Question\s*\n+(.*?)\n+### Answer", re.S)
ANSWER_SECTION = re.compile(r"### Answer\s*\n+(.*?)\s*$", re.S)
TRAILING_RULE = re.compile(r"\n+-{3,}\s*$")


class EmptyDocumentError(ValueError):
    """Raised when an upload has nothing that can be indexed."""


def load_markdown_records(path: Path) -> list[Document]:
    """Parse the IT Support Q&A Markdown format into one Document per record.

    The source file keeps the file path in ``source`` so replacement and the
    BM25 index still group all records from the same document.
    """
    text = path.read_text(encoding="utf-8")
    headings = list(RECORD_HEADING.finditer(text))
    if not headings:
        return []

    documents = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        block = text[heading.start():end]
        question_match = QUESTION_SECTION.search(block)
        answer_match = ANSWER_SECTION.search(block)
        if not question_match or not answer_match:
            continue

        question = question_match.group(1).strip()
        answer = TRAILING_RULE.sub("", answer_match.group(1)).strip()
        if not question or not answer:
            continue

        meta = dict(METADATA_LINE.findall(block))
        record_number = meta.get("record_number", "")
        metadata = {
            "source": str(path),
            "record_id": meta.get("id") or heading.group(1),
            "question": question,
            "dataset_source": meta.get("source", ""),
            "category": meta.get("category", ""),
            "file": meta.get("file", path.name),
            "document_type": meta.get("document_type", ""),
            "language": meta.get("language", ""),
            "start_index": heading.start(),
        }
        if record_number.isdigit():
            metadata["record_number"] = int(record_number)
        documents.append(
            Document(page_content=f"{question}\n\n{answer}", metadata=metadata)
        )
    return documents


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


def index_file(path: Path) -> tuple[int, list[str]]:
    """Parse, chunk, embed and store one document. Used by inline and queued ingestion."""
    chunks = chunk_documents(load_file(path))
    if not chunks:
        raise EmptyDocumentError("The uploaded document contains no indexable text")
    return len(chunks), replace_documents(chunks)

"""Build the BM25 corpus from bundled documents without calling Pinecone.

The corpus is a deployment artifact (it is gitignored), so the Docker build
regenerates it from ``data/sample_kb``. No embedding model is loaded.
"""

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.rag.hybrid import Bm25Index
from app.services.ingestion import SUPPORTED, chunk_documents, load_file


def build(directory: Path, output: Path) -> int:
    documents = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            documents.extend(chunk_documents(load_file(path)))
    if not documents:
        raise ValueError(f"No supported documents in {directory}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    Bm25Index(output).add(documents)
    return len(documents)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Build the BM25 corpus from a directory of documents."
    )
    parser.add_argument("directory", nargs="?", default=settings.sample_kb_dir, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path(settings.bm25_index_path)
    )
    args = parser.parse_args(argv)
    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")
    count = build(args.directory, args.output)
    print(f"{args.output}: {count} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError(f"not a directory: {directory}")
    if output.is_symlink():
        raise ValueError(f"refusing to write through a symlink: {output}")
    output = output.parent.resolve() / output.name
    if output == directory or directory in output.parents:
        raise ValueError("BM25 output must be outside the source directory")

    documents = []
    for path in sorted(directory.iterdir()):
        if not path.is_symlink() and path.is_file() and path.suffix.lower() in SUPPORTED:
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
    source_root = Path(settings.sample_kb_dir).resolve()
    output_root = Path(settings.bm25_index_path).parent.resolve()
    directory = args.directory.resolve()
    output = args.output
    output_parent = output.parent.resolve()
    if directory != source_root and source_root not in directory.parents:
        parser.error(f"source directory must be within {source_root}")
    if output.is_symlink() or output_parent != output_root:
        parser.error(f"output must be a regular file within {output_root}")
    output = output_parent / output.name
    try:
        count = build(directory, output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{output}: {count} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

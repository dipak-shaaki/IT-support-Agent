"""Index every supported document in a directory.

Run from the repository root:
    uv run --env-file .env python -m app.bulk_ingest data/sample_kb --dry-run
    uv run --env-file .env python -m app.bulk_ingest data/sample_kb
"""

import argparse
import sys
from pathlib import Path

from app.core.logging import configure_logging
from app.services.ingestion import SUPPORTED, chunk_documents, index_file, load_file


def run(directory: Path, dry_run: bool = False) -> int:
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED
    )
    if not files:
        print(f"No supported files in {directory}")
        return 1

    failures = 0
    for path in files:
        try:
            if dry_run:
                chunks = chunk_documents(load_file(path))
                print(f"{path.name}: {len(chunks)} chunks (dry run)")
            else:
                chunks, ids = index_file(path)
                print(f"{path.name}: {chunks} chunks, {len(ids)} vectors")
        except Exception as exc:
            failures += 1
            print(f"{path.name}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"Done: {len(files) - failures}/{len(files)} files processed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Index a directory of IT knowledge-base documents."
    )
    parser.add_argument("directory", nargs="?", default="data/sample_kb", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk without writing to Pinecone or BM25.",
    )
    args = parser.parse_args(argv)
    configure_logging()
    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")
    return run(args.directory, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

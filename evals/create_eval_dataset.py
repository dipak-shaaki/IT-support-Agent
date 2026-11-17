import argparse
import json
import random
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_DIR = Path("data/sample_kb")
OUTPUT_FILE = Path("evals/eval_dataset.json")

SAMPLE_SIZE = 15
RANDOM_SEED = 42


# --------------------------------------------------
# Extract metadata from a Markdown record
# --------------------------------------------------

def extract_metadata(text):
    metadata = {}

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            key, separator, value = stripped[2:].partition(":")
            if separator:
                key = key.strip()
                value = value.strip()
                metadata[key] = value

    return metadata


def _is_record_heading(line):
    parts = line.strip().split()
    if len(parts) != 2 or parts[0] != "##":
        return False
    prefix, separator, number = parts[1].partition("-")
    return bool(separator and prefix.isalpha() and prefix.isupper() and number.isdigit())


# --------------------------------------------------
# Extract content between Markdown headings
# --------------------------------------------------

def extract_section(text, heading, next_heading=None):
    lines = text.splitlines()
    heading_line = f"### {heading}"
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == heading_line),
        None,
    )
    if start is None:
        return ""

    end = len(lines)
    if next_heading:
        next_heading_line = f"### {next_heading}"
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if lines[index].strip() == next_heading_line
            ),
            len(lines),
        )
    return "\n".join(lines[start + 1 : end]).strip().strip("-").strip()


def _remove_metadata_section(lines):
    result = []
    skip_metadata = False
    for line in lines:
        stripped = line.strip()
        if stripped == "### Metadata":
            skip_metadata = True
            continue
        if skip_metadata and stripped.startswith("### "):
            skip_metadata = False
        if not skip_metadata:
            result.append(line)
    return result


# --------------------------------------------------
# Parse one record
# --------------------------------------------------

def parse_record(block, file_path, index):
    metadata = extract_metadata(block)

    question = extract_section(
        block,
        "Question",
        "Answer"
    )

    answer = extract_section(
        block,
        "Answer"
    )

    # Handle records without a ### Question heading.
    # Example: question text appears before ### Answer.
    if not question and answer:
        lines = block.splitlines()
        answer_heading = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip() == "### Answer"
            ),
            len(lines),
        )
        question_lines = _remove_metadata_section(lines[:answer_heading])
        question = "\n".join(
            line for line in question_lines if not _is_record_heading(line)
        ).strip()

    if not question or not answer:
        return None

    record_id = metadata.get("id", f"UNASSIGNED_{file_path.stem}_{index:04d}")

    return {
        "id": record_id,
        "question": question,
        "reference_answer": answer,
        "source": metadata.get("source", ""),
        "category": metadata.get("category", ""),
        "file": metadata.get("file", file_path.name),
        "record_number": metadata.get("record_number", ""),
        "document_type": metadata.get("document_type", ""),
        "language": metadata.get("language", "")
    }


# --------------------------------------------------
# Read all Markdown files
# --------------------------------------------------

def load_dataset(input_dir):
    records = []

    markdown_files = sorted(input_dir.rglob("*.md"))

    if not markdown_files:
        raise FileNotFoundError(
            f"No Markdown files found in {input_dir.resolve()}"
        )

    for file_path in markdown_files:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)
        record_starts = [
            index for index, line in enumerate(lines) if _is_record_heading(line)
        ]
        if not record_starts:
            blocks = [content]
        else:
            starts = [0, *record_starts]
            ends = [*record_starts, len(lines)]
            blocks = [
                "".join(lines[start:end])
                for start, end in zip(starts, ends)
                if start < end
            ]

        for index, block in enumerate(blocks, start=1):
            if not block.strip():
                continue

            record = parse_record(block, file_path, index)

            if record:
                records.append(record)

    return records


# --------------------------------------------------
# Select and save evaluation dataset
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create a random QA evaluation dataset from KB markdown files.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="Directory containing markdown KB files.")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Output JSON file path.")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help="Number of questions to sample (12-15 recommended).")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducible sampling.")
    args = parser.parse_args()

    dataset = load_dataset(args.input_dir)

    print(f"Total valid QA records found: {len(dataset)}")

    if len(dataset) < args.sample_size:
        raise ValueError(
            f"Need {args.sample_size} records, "
            f"but found only {len(dataset)}."
        )

    rng = random.Random(args.seed)  # NOSONAR python:S2245 - reproducible non-security sampling

    evaluation_set = rng.sample(dataset, args.sample_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(
            evaluation_set,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"Selected: {len(evaluation_set)} questions")
    print(f"Output: {args.output.resolve()}")

    print("\nSelected record IDs:")
    for record in evaluation_set:
        print(f"- {record['id']}")


if __name__ == "__main__":
    main()

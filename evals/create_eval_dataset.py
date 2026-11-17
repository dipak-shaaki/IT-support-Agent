import argparse
import json
import random
import re
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
        match = re.match(r"^\s*-\s*([^:]+):\s*(.*)", line)

        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            metadata[key] = value

    return metadata


# --------------------------------------------------
# Extract content between Markdown headings
# --------------------------------------------------

def extract_section(text, heading, next_heading=None):
    pattern = rf"(?ms)^###\s+{re.escape(heading)}\s*$\n(.*?)"

    if next_heading:
        pattern += rf"(?=^###\s+{re.escape(next_heading)}\s*$|\Z)"
    else:
        pattern += r"(?=\Z)"

    match = re.search(pattern, text)

    if not match:
        return ""

    return match.group(1).strip().strip("-").strip()


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
        before_answer = re.split(
            r"(?m)^###\s+Answer\s*$",
            block,
            maxsplit=1
        )[0]

        # Remove metadata section if present
        before_answer = re.sub(
            r"(?ms)^###\s+Metadata\s*\n.*?(?=(?:^###\s|\Z))",
            "",
            before_answer
        )

        # Remove record ID heading
        before_answer = re.sub(
            r"(?m)^##\s+[A-Z]+-\d+\s*$",
            "",
            before_answer
        )

        question = before_answer.strip()

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

        # Split when a new record starts, e.g. ## IT-00005
        blocks = re.split(
            r"(?m)(?=^##\s+[A-Z]+-\d+\s*$)",
            content
        )

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

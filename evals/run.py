import argparse
import hashlib
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from langchain_core.messages import ToolMessage

from app.core.config import get_settings
from app.agent.graph import ask

CASES_PATH = Path(__file__).with_name("cases.jsonl")
logger = logging.getLogger(__name__)


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    cases = []
    seen_names = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        required = {"name", "question", "expected_source"}
        missing = required - case.keys()
        if missing:
            raise ValueError(
                f"{path}:{line_number}: missing fields {', '.join(sorted(missing))}"
            )
        if case["name"] in seen_names:
            raise ValueError(f"{path}:{line_number}: duplicate case name {case['name']!r}")
        seen_names.add(case["name"])
        for field in ("expected_tools", "expected_relevant_sources"):
            if field in case and (
                not isinstance(case[field], list)
                or not all(isinstance(value, str) for value in case[field])
            ):
                raise ValueError(
                    f"{path}:{line_number}: {field} must be a list of strings"
                )
        cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def _called_tools(result: dict[str, Any]) -> list[str]:
    return [
        line.split("Tool call → ", 1)[1].split("(", 1)[0]
        for line in result.get("trace", [])
        if line.startswith("Tool call → ")
    ]


def _retrieved_documents(result: dict[str, Any]) -> list[dict[str, str]]:
    documents = []
    for message in result.get("messages", []):
        if not isinstance(message, ToolMessage) or message.name != "search_private_kb":
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            continue
        documents.extend(
            {
                "source": str(document.get("source", "")),
                "content": str(document.get("content", "")),
            }
            for document in payload.get("documents", [])
            if isinstance(document, dict)
        )
    return documents


def _source_name(source: str) -> str:
    return PurePosixPath(source.replace("\\", "/")).name.casefold()


def _report_output_path(path: Path) -> Path:
    report_root = Path(__file__).with_name("reports").resolve()
    if path.is_symlink():
        raise ValueError(f"Refusing to write evaluation output through a symlink: {path}")
    resolved = path.resolve()
    if resolved == report_root or report_root not in resolved.parents:
        raise ValueError(f"Evaluation output must be within {report_root}")
    return resolved


def _retrieval_scores(
    documents: list[dict[str, str]], expected_sources: list[str], k: int = 5
) -> dict[str, float]:
    expected = {_source_name(source) for source in expected_sources}
    ranked_sources = []
    seen_sources = set()
    for document in documents:
        source = _source_name(document["source"])
        if source and source not in seen_sources:
            ranked_sources.append(source)
            seen_sources.add(source)

    relevant_found = [source for source in ranked_sources if source in expected]
    first_relevant_rank = next(
        (
            rank
            for rank, source in enumerate(ranked_sources, start=1)
            if source in expected
        ),
        0,
    )
    top_k_sources = ranked_sources[:k]
    relevant_in_top_k = sum(source in expected for source in top_k_sources)
    first_relevant_rank_at_k = next(
        (
            rank
            for rank, source in enumerate(top_k_sources, start=1)
            if source in expected
        ),
        0,
    )
    return {
        "precision": len(relevant_found) / len(ranked_sources) if ranked_sources else 0.0,
        "recall": len(relevant_found) / len(expected) if expected else 0.0,
        "mrr": 1 / first_relevant_rank if first_relevant_rank else 0.0,
        "precision_at_5": relevant_in_top_k / k,
        "recall_at_5": relevant_in_top_k / len(expected) if expected else 0.0,
        "hit_rate_at_5": float(relevant_in_top_k > 0),
        "mrr_at_5": 1 / first_relevant_rank_at_k if first_relevant_rank_at_k else 0.0,
    }


def _normalize(text: str) -> str:
    # Match expected phrases regardless of typographic quotes/dashes from models.
    return text.translate(
        str.maketrans({
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u00a0": " ",
        })
    ).casefold()


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    tools = _called_tools(result)
    answer = _normalize(result.get("answer", ""))
    documents = _retrieved_documents(result)
    evidence = _normalize("\n".join(document["content"] for document in documents))
    citation_types = {citation.get("type") for citation in result.get("citations", [])}
    retrieval_scores = None
    expected_sources = case.get("expected_relevant_sources")
    if expected_sources is not None:
        retrieval_scores = _retrieval_scores(documents, expected_sources)
    checks = {
        "expected_source": result.get("source_used") == case["expected_source"],
        "expected_tools": (
            "expected_tools" not in case or tools == case["expected_tools"]
        ),
        "required_tools": all(tool in tools for tool in case.get("required_tools", [])),
        "tool_order": [
            tools.index(tool) for tool in case.get("required_tools", []) if tool in tools
        ]
        == sorted(
            tools.index(tool) for tool in case.get("required_tools", []) if tool in tools
        ),
        "forbidden_tools": not any(
            tool in tools for tool in case.get("forbidden_tools", [])
        ),
        "required_citation_type": (
            case.get("required_citation_type") is None
            or case["required_citation_type"] in citation_types
        ),
        "evidence_must_include": all(
            _normalize(phrase) in evidence
            for phrase in case.get("evidence_must_include", [])
        ),
        "answer_must_include": all(
            _normalize(phrase) in answer for phrase in case.get("answer_must_include", [])
        ),
        "answer_must_include_any": (
            not case.get("answer_must_include_any")
            or any(
                _normalize(phrase) in answer
                for phrase in case["answer_must_include_any"]
            )
        ),
        "retrieval_relevant_source": (
            retrieval_scores is None
            or retrieval_scores["recall"] > 0
            or not expected_sources
        ),
    }
    return {
        "name": case["name"],
        "passed": all(checks.values()),
        "checks": checks,
        "source_used": result.get("source_used"),
        "tools_called": tools,
        "citation_types": sorted(item for item in citation_types if item),
        "evidence_checked": bool(case.get("evidence_must_include")),
        "retrieval": retrieval_scores,
        "answer": result.get("answer", ""),
    }


def _rate(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def run_evaluations(
    cases: list[dict[str, Any]], repeats: int = 1
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    started_at = datetime.now(timezone.utc)
    results = []
    latencies = []
    for repetition in range(1, repeats + 1):
        for case in cases:
            started = time.perf_counter()
            try:
                result = ask(case["question"])
                evaluated = evaluate_case(case, result)
            except Exception as exc:
                logger.error("Evaluation case %s failed: %s", case["name"], exc)
                evaluated = {
                    "name": case["name"],
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            evaluated["repetition"] = repetition
            evaluated["latency_seconds"] = elapsed
            results.append(evaluated)

    passed = sum(result["passed"] for result in results)
    source_cases = []
    tool_cases = []
    retrieval_cases = []
    for repetition in range(repeats):
        start = repetition * len(cases)
        repeat_results = results[start : start + len(cases)]
        for case, result in zip(cases, repeat_results):
            if "expected_source" in case:
                source_cases.append((case, result))
            if "expected_tools" in case:
                tool_cases.append((case, result))
            if "expected_relevant_sources" in case:
                retrieval_cases.append((case, result))

    settings = get_settings()
    return {
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results) if results else 0.0,
        "run_metadata": {
            "started_at_utc": started_at.isoformat(),
            "repeats": repeats,
            "cases_per_repeat": len(cases),
            "retrieval_top_k": settings.top_k,
            "model": settings.groq_model,
            "embedding_model": settings.embedding_model,
            "latency_percentile_method": "nearest_rank",
        },
        "metrics": {
            "source_accuracy": _rate(
                sum(
                    result.get("source_used") == case["expected_source"]
                    for case, result in source_cases
                ),
                len(source_cases),
            ),
            "tool_selection_accuracy": _rate(
                sum(
                    result.get("tools_called", []) == case["expected_tools"]
                    for case, result in tool_cases
                ),
                len(tool_cases),
            ),
            "retrieval_precision": _rate(
                sum(
                    (result.get("retrieval") or {}).get("precision", 0.0)
                    for _, result in retrieval_cases
                ),
                len(retrieval_cases),
            ),
            "retrieval_recall": _rate(
                sum(
                    (result.get("retrieval") or {}).get("recall", 0.0)
                    for _, result in retrieval_cases
                ),
                len(retrieval_cases),
            ),
            "retrieval_mrr": _rate(
                sum(
                    (result.get("retrieval") or {}).get("mrr", 0.0)
                    for _, result in retrieval_cases
                ),
                len(retrieval_cases),
            ),
            "retrieval_precision_at_5": _rate(
                sum(
                    (result.get("retrieval") or {}).get("precision_at_5", 0.0)
                    for _, result in retrieval_cases
                ),
                len(retrieval_cases),
            ),
            "retrieval_recall_at_5": _rate(
                sum(
                    (result.get("retrieval") or {}).get("recall_at_5", 0.0)
                    for _, result in retrieval_cases
                ),
                len(retrieval_cases),
            ),
            "retrieval_hit_rate_at_5": _rate(
                sum(
                    (result.get("retrieval") or {}).get("hit_rate_at_5", 0.0)
                    for _, result in retrieval_cases
                ),
                len(retrieval_cases),
            ),
            "retrieval_mrr_at_5": _rate(
                sum(
                    (result.get("retrieval") or {}).get("mrr_at_5", 0.0)
                    for _, result in retrieval_cases
                ),
                len(retrieval_cases),
            ),
            "retrieval_cases": len(retrieval_cases),
            "tool_selection_cases": len(tool_cases),
            "latency_samples": len(latencies),
            "latency_mean_seconds": _rate(sum(latencies), len(latencies)),
            "latency_p95_seconds": _percentile(latencies, 0.95),
            "latency_p99_seconds": _percentile(latencies, 0.99),
        },
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the live IT support agent evaluation.")
    parser.add_argument("--repeats", type=int, default=1, help="Run each case this many times.")
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    cases = load_cases()
    report = run_evaluations(cases, repeats=args.repeats)
    report["run_metadata"]["dataset_sha256"] = hashlib.sha256(
        CASES_PATH.read_bytes()
    ).hexdigest()
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        try:
            report_path = _report_output_path(args.output)
        except ValueError as exc:
            parser.error(str(exc))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(output + "\n", encoding="utf-8")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())

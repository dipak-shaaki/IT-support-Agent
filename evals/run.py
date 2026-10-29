import json
import logging
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage

from app.rag.workflow import ask

CASES_PATH = Path(__file__).with_name("cases.jsonl")
logger = logging.getLogger(__name__)


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    cases = []
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


def _retrieved_evidence(result: dict[str, Any]) -> str:
    evidence = []
    for message in result.get("messages", []):
        if not isinstance(message, ToolMessage) or message.name != "search_private_kb":
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            continue
        evidence.extend(
            document.get("content", "")
            for document in payload.get("documents", [])
            if isinstance(document, dict)
        )
    return "\n".join(evidence).casefold()


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    tools = _called_tools(result)
    answer = result.get("answer", "").casefold()
    evidence = _retrieved_evidence(result)
    citation_types = {citation.get("type") for citation in result.get("citations", [])}
    checks = {
        "expected_source": result.get("source_used") == case["expected_source"],
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
            phrase.casefold() in evidence
            for phrase in case.get("evidence_must_include", [])
        ),
        "answer_must_include": all(
            phrase.casefold() in answer for phrase in case.get("answer_must_include", [])
        ),
        "answer_must_include_any": (
            not case.get("answer_must_include_any")
            or any(
                phrase.casefold() in answer
                for phrase in case["answer_must_include_any"]
            )
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
        "answer": result.get("answer", ""),
    }


def run_evaluations(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for case in cases:
        try:
            result = ask(case["question"])
            results.append(evaluate_case(case, result))
        except Exception as exc:
            logger.exception("Evaluation case %s failed", case["name"])
            results.append(
                {
                    "name": case["name"],
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    passed = sum(result["passed"] for result in results)
    return {
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results) if results else 0.0,
        "cases": results,
    }


def main() -> int:
    report = run_evaluations(load_cases())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())

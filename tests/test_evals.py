from evals import run as eval_runner
from evals.run import evaluate_case, load_cases
from langchain_core.messages import ToolMessage


def test_eval_cases_define_expected_behavior():
    cases = load_cases()

    assert {case["name"] for case in cases} == {
        "vpn_access",
        "password_reset",
        "direct_greeting",
        "current_vendor_status",
        "unknown_personal_device_fact",
    }


def test_evaluator_checks_source_tooling_citations_and_answer():
    case = {
        "name": "vpn",
        "expected_source": "private_kb",
        "required_tools": ["search_private_kb"],
        "forbidden_tools": ["search_web"],
        "required_citation_type": "private_kb",
        "evidence_must_include": ["NovaSecure"],
        "answer_must_include": ["NovaSecure", "MFA"],
    }
    result = {
        "source_used": "private_kb",
        "trace": ["Tool call → search_private_kb(vpn)", "Private KB results → 1 chunks"],
        "citations": [{"type": "private_kb"}],
        "messages": [
            ToolMessage(
                name="search_private_kb",
                tool_call_id="kb-1",
                content='{"documents":[{"content":"Install NovaSecure and complete MFA."}]}',
            )
        ],
        "answer": "Install NovaSecure and complete MFA.",
    }

    evaluated = evaluate_case(case, result)

    assert evaluated["passed"] is True
    assert all(evaluated["checks"].values())


def test_evaluator_fails_when_retrieved_evidence_is_not_relevant():
    case = {
        "name": "vpn",
        "expected_source": "private_kb",
        "evidence_must_include": ["NovaSecure"],
    }
    result = {
        "source_used": "private_kb",
        "trace": ["Tool call → search_private_kb(vpn)"],
        "messages": [
            ToolMessage(
                name="search_private_kb",
                tool_call_id="kb-1",
                content='{"documents":[{"content":"Contact the help desk."}]}',
            )
        ],
        "answer": "Use the VPN client.",
    }

    evaluated = evaluate_case(case, result)

    assert evaluated["checks"]["evidence_must_include"] is False
    assert evaluated["passed"] is False


def test_evaluator_fails_when_tool_order_or_answer_are_wrong():
    case = {
        "name": "web_fallback",
        "expected_source": "web_search",
        "required_tools": ["search_private_kb", "search_web"],
        "required_citation_type": "web",
    }
    result = {
        "source_used": "web_search",
        "trace": [
            "Tool call → search_web(outage)",
            "Tool call → search_private_kb(outage)",
        ],
        "citations": [{"type": "web"}],
        "answer": "External information.",
    }

    evaluated = evaluate_case(case, result)

    assert evaluated["checks"]["required_tools"] is True
    assert evaluated["checks"]["tool_order"] is False
    assert evaluated["passed"] is False


def test_provider_failure_counts_as_failed_case(monkeypatch):
    monkeypatch.setattr(
        eval_runner,
        "ask",
        lambda question: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )

    report = eval_runner.run_evaluations(
        [{"name": "provider_error", "question": "test", "expected_source": "direct"}]
    )

    assert report["passed"] == 0
    assert report["total"] == 1
    assert report["cases"][0]["passed"] is False
    assert report["cases"][0]["error"] == "RuntimeError: provider unavailable"

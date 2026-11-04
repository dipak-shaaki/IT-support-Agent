from evals import run as eval_runner
from evals.run import evaluate_case, load_cases
from langchain_core.messages import ToolMessage


def test_eval_cases_define_expected_behavior():
    cases = load_cases()

    assert len(cases) >= 40
    assert {
        "vpn_access",
        "password_reset",
        "mfa_policy",
        "software_installation",
        "lost_laptop",
        "laptop_performance",
        "direct_greeting",
        "current_vendor_status",
        "external_company_policy",
        "unknown_personal_device_fact",
    } <= {case["name"] for case in cases}
    assert sum("expected_relevant_sources" in case for case in cases) >= 35


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


def test_evaluator_calculates_source_level_retrieval_metrics():
    case = {
        "name": "vpn",
        "expected_source": "private_kb",
        "expected_tools": ["search_private_kb"],
        "expected_relevant_sources": ["company_it_handbook.md"],
    }
    result = {
        "source_used": "private_kb",
        "trace": ["Tool call → search_private_kb(vpn)"],
        "citations": [{"type": "private_kb"}],
        "messages": [
            ToolMessage(
                name="search_private_kb",
                tool_call_id="kb-1",
                content=(
                    '{"documents":['
                    '{"source":"/kb/company_it_handbook.md","content":"VPN access"},'
                    '{"source":"/kb/company_it_handbook.md","content":"MFA required"},'
                    '{"source":"/kb/service_desk_runbook.md","content":"Email"}]}'
                ),
            )
        ],
        "answer": "Use the VPN client.",
    }

    evaluated = evaluate_case(case, result)

    assert evaluated["passed"] is True
    assert evaluated["retrieval"] == {
        "precision": 0.5,
        "recall": 1.0,
        "mrr": 1.0,
        "precision_at_5": 0.2,
        "recall_at_5": 1.0,
        "hit_rate_at_5": 1.0,
        "mrr_at_5": 1.0,
    }


def test_evaluator_reports_aggregate_retrieval_and_tool_metrics(monkeypatch):
    results = [
        {
            "source_used": "private_kb",
            "trace": ["Tool call → search_private_kb(vpn)"],
            "citations": [{"type": "private_kb"}],
            "messages": [
                ToolMessage(
                    name="search_private_kb",
                    tool_call_id="kb-1",
                    content=(
                        '{"documents":[{"source":"company_it_handbook.md",'
                        '"content":"VPN"}]}'
                    ),
                )
            ],
            "answer": "Use the VPN client.",
        },
        {
            "source_used": "direct",
            "trace": [],
            "citations": [],
            "messages": [],
            "answer": "Hello!",
        },
    ]
    monkeypatch.setattr(eval_runner, "ask", lambda question: results.pop(0))
    cases = [
        {
            "name": "retrieval",
            "question": "How do I connect to VPN?",
            "expected_source": "private_kb",
            "expected_tools": ["search_private_kb"],
            "expected_relevant_sources": ["company_it_handbook.md"],
        },
        {
            "name": "greeting",
            "question": "Hello!",
            "expected_source": "direct",
            "expected_tools": [],
        },
    ]

    report = eval_runner.run_evaluations(cases)

    assert report["pass_rate"] == 1.0
    expected_metrics = {
        "source_accuracy": 1.0,
        "tool_selection_accuracy": 1.0,
        "retrieval_precision": 1.0,
        "retrieval_recall": 1.0,
        "retrieval_mrr": 1.0,
        "retrieval_precision_at_5": 0.2,
        "retrieval_recall_at_5": 1.0,
        "retrieval_hit_rate_at_5": 1.0,
        "retrieval_mrr_at_5": 1.0,
        "retrieval_cases": 1,
        "tool_selection_cases": 2,
        "latency_samples": 2,
    }
    assert {
        key: value
        for key, value in report["metrics"].items()
        if not key.startswith("latency_") or key == "latency_samples"
    } == expected_metrics
    assert report["run_metadata"]["repeats"] == 1
    assert report["run_metadata"]["cases_per_repeat"] == 2
    assert report["metrics"]["latency_mean_seconds"] >= 0
    assert report["metrics"]["latency_p95_seconds"] >= 0
    assert report["metrics"]["latency_p99_seconds"] >= 0


def test_retrieval_at_five_uses_unique_sources_and_expected_labels():
    documents = [
        {"source": "a.md", "content": "first chunk"},
        {"source": "a.md", "content": "second chunk"},
        {"source": "b.md", "content": "second source"},
        {"source": "c.md", "content": "third source"},
        {"source": "d.md", "content": "fourth source"},
        {"source": "e.md", "content": "fifth source"},
        {"source": "f.md", "content": "sixth source"},
    ]

    scores = eval_runner._retrieval_scores(documents, ["d.md", "f.md"])

    assert scores["precision_at_5"] == 0.2
    assert scores["recall_at_5"] == 0.5
    assert scores["hit_rate_at_5"] == 1.0
    assert scores["mrr_at_5"] == 0.25


def test_percentile_uses_nearest_rank():
    assert eval_runner._percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0
    assert eval_runner._percentile([], 0.95) is None


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
    assert report["metrics"]["source_accuracy"] == 0.0
    assert report["metrics"]["tool_selection_accuracy"] is None

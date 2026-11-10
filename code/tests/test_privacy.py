from app.core.privacy import redact_pii


def test_redact_pii_removes_identity_contact_and_secret_values():
    text = (
        "user alice@example.com phone +1 (555) 123-4567 "
        "key sk-abcdefghijklmnopqrstuvwxyz012345"
    )

    redacted = redact_pii(text)

    assert "alice@example.com" not in redacted
    assert "555" not in redacted
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in redacted
    assert redacted.count("[redacted]") == 3


def test_redact_pii_keeps_ordinary_support_queries():
    assert redact_pii("Is Microsoft Teams down right now?") == (
        "Is Microsoft Teams down right now?"
    )

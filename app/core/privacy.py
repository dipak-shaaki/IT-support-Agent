import re

# shortcut: regex redaction is best-effort; add a DLP service if policy requires it
_PATTERNS = (
    re.compile(r"\b[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+\b"),
    re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)"),
    re.compile(r"\b(?:sk|pk|ghp|gho|github_pat|xox[baprs])[-_][-A-Za-z0-9_]{16,}\b", re.I),
    re.compile(r"\b[-A-Za-z0-9_]{32,}\b"),
)


def redact_pii(text: str) -> str:
    redacted = text
    for pattern in _PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    return redacted

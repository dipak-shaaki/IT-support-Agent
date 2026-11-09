import re

from langchain_core.messages import AnyMessage, HumanMessage


MAX_CONSECUTIVE_ACKNOWLEDGMENTS = 4
ACKNOWLEDGMENT_WORDS = {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "right"}
ACKNOWLEDGMENT_PHRASES = {"got it", "understood", "thanks", "thank you"}

CURRENT_STATUS_PATTERN = re.compile(
    r"\b("
    r"(?:is|are)\b.{0,40}\b(?:down|having|experiencing)\b"
    r"|any (?:current )?outages?"
    r"|is there an? (?:current )?outage"
    r"|(?:has|have|was|were) there .{0,20}outages?"
    r"|service status|status page|service health|currently unavailable"
    r"|having issues right now|service disruption"
    r")",
    re.IGNORECASE,
)


def is_acknowledgment_only(text: str) -> bool:
    normalized = " ".join(re.sub(r"[\W_]+", " ", text.casefold()).split())
    if normalized in ACKNOWLEDGMENT_PHRASES:
        return True
    words = normalized.split()
    return bool(words) and all(word in ACKNOWLEDGMENT_WORDS for word in words)


def has_repetitive_follow_up(
    question: str, history: list[AnyMessage]
) -> bool:
    if not is_acknowledgment_only(question):
        return False

    consecutive = 1
    for message in reversed(history):
        if not isinstance(message, HumanMessage):
            continue
        if not isinstance(message.content, str) or not is_acknowledgment_only(message.content):
            break
        consecutive += 1
        if consecutive >= MAX_CONSECUTIVE_ACKNOWLEDGMENTS:
            return True
    return False

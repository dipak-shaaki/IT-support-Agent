import re

from langchain_core.messages import AnyMessage, HumanMessage


MAX_CONSECUTIVE_ACKNOWLEDGMENTS = 4
ACKNOWLEDGMENT_WORDS = {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "right"}
ACKNOWLEDGMENT_PHRASES = {"got it", "understood", "thanks", "thank you"}
BRIEF_PLEASANTRY_PATTERN = re.compile(
    r"(?:hi|hello|hey|hiya|howdy)(?:\s+(?:again|there|everyone))?"
    r"|thanks?(?:\s+again)?|thank you(?:\s+again)?|thx"
    r"|good morning|good afternoon|good evening",
    re.IGNORECASE,
)
ANSWER_FEEDBACK_PATTERN = re.compile(
    r"^\s*(?:your\s+(?:answers?|responses?|replies|information)|"
    r"(?:this|that)\s+(?:answers?|responses?|replies|information))\b.{0,120}"
    r"\b(?:wrong|incorrect|outdated|not\s+up[\s-]?to[\s-]?date)\b"
    r"|^\s*(?:this|that|it)\s+(?:is|seems|looks)\s+"
    r"(?:wrong|incorrect|outdated|not\s+up[\s-]?to[\s-]?date)\b"
    r"|\bwrong\s+(?:documents?|docs?|resources?|citations?)\b",
    re.IGNORECASE,
)
ANSWER_SOURCE_FOLLOWUP_PATTERN = re.compile(
    r"^\s*(?:(?:is|was)\s+)?(?:this|that|it|your\s+(?:answer|response))"
    r".{0,100}\b(?:from|source|citation|document|resource|company\s+(?:kb|knowledge\s+base))\b"
    r"|\bwhere\s+did\s+you\s+get\s+(?:this|that|it|the\s+(?:answer|information))\b"
    r"|\bshow\s+me\s+(?:the\s+)?(?:source|citation|document)\b",
    re.IGNORECASE,
)

CURRENT_STATUS_PATTERN = re.compile(
    r"\b("
    r"(?:is|are)\b.{0,40}\b(?:down|having|experiencing)\b"
    r"|any (?:current )?outages?"
    r"|is there an? (?:current )?outage"
    r"|(?:has|have|was|were) there .{0,20}outages?"
    r"|(?:latest|current|recent|ongoing|active) .{0,30}outages?"
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


def is_brief_pleasantry(text: str) -> bool:
    return bool(
        BRIEF_PLEASANTRY_PATTERN.fullmatch(text.strip().rstrip("!?. ,"))
    )


def is_answer_feedback(text: str) -> bool:
    return bool(ANSWER_FEEDBACK_PATTERN.search(text))


def is_answer_source_followup(text: str) -> bool:
    return bool(ANSWER_SOURCE_FOLLOWUP_PATTERN.search(text))


def is_non_retrieval_turn(text: str) -> bool:
    return (
        is_brief_pleasantry(text)
        or is_acknowledgment_only(text)
        or (
            is_answer_feedback(text)
            and not is_answer_source_followup(text)
        )
    )


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

"""Guardrails: prompt-injection detection and PII masking.

Injection checks run twice: on the user's question (direct injection) and on every retrieved
chunk (indirect injection - instructions hidden inside a document). Flagged chunks are dropped
before they reach any LLM.
"""

import re
from dataclasses import dataclass

_INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (all |any |the )?(previous|prior|above|system) (instructions|prompts|rules)",
    r"forget (all |your )?(previous |prior )?(instructions|rules)",
    r"(reveal|print|show|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now (a|an|in)\b",
    r"\b(developer|dan|jailbreak) mode\b",
    r"<\s*/?\s*system\s*>",
    r"begin (system|new) (prompt|instructions)",
    r"(send|email|forward|upload) (all|every|the) (documents|files|data|records) to",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMIRATES_ID": re.compile(r"\b784-?\d{4}-?\d{7}-?\d\b"),
    "IBAN": re.compile(r"\bAE\d{2}\s?(?:\d{4}\s?){4}\d{3}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"(?:\+971|00971|\b0)\s?-?(?:5\d|[2-9])\s?-?\d{3}\s?-?\d{4}\b"),
}


@dataclass(frozen=True)
class InjectionVerdict:
    flagged: bool
    matched: str = ""


def detect_injection(text: str) -> InjectionVerdict:
    match = _INJECTION_RE.search(text)
    return InjectionVerdict(True, match.group(0)) if match else InjectionVerdict(False)


def strip_injections(text: str) -> tuple[str, list[str]]:
    """Remove only the sentences carrying injected instructions, keeping the document's legitimate content.

    Returns the cleaned text and the removed fragments.
    """
    removed: list[str] = []
    kept_lines = []
    for line in text.splitlines():
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if _INJECTION_RE.search(sentence):
                removed.append(sentence.strip())
            else:
                kept.append(sentence)
        kept_lines.append(" ".join(kept))
    return "\n".join(kept_lines), removed


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII with typed placeholders. Returns the masked text and the PII types found."""
    found: list[str] = []
    for label, pattern in _PII_PATTERNS.items():
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            found.append(label)
    return text, found

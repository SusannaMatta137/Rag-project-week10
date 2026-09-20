# compliance.py
# -------------
# Week 18: metadata tagging and automated redaction.
#
# This module labels text by sensitivity and data type, then redacts
# common PII/financial patterns before logs, analytics, or LLM calls.
# Classification is rule-based, not perfect — the goal is consistent
# labeling and safe defaults at system boundaries.

import re

# --- Metadata vocabulary ---
SENSITIVITY_LEVELS = ("public", "internal", "confidential", "restricted")
DATA_TYPES = ("operational", "PII", "PHI", "financial")
SOURCES = ("user_input", "document", "model_output")

# Patterns used for both tagging and redaction.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_EMPLOYEE_ID_RE = re.compile(r"\bEMP-\d+\b", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"\b(?:api[_-]?key|secret|token|password|bearer)\s*[:=]\s*\S+",
    re.IGNORECASE,
)

_PHI_KEYWORDS = (
    "diagnosis",
    "medical record",
    "patient",
    "prescription",
    "hipaa",
    "health insurance",
)

_INTERNAL_MARKERS = ("internal note", "internal only", "confidential", "do not share")


def _detected_types(text):
    """Return a list of data_type labels found in text."""
    if not text:
        return ["operational"]

    lowered = text.lower()
    types = []

    if _EMAIL_RE.search(text) or _PHONE_RE.search(text) or _SSN_RE.search(text) or _EMPLOYEE_ID_RE.search(text):
        types.append("PII")
    if _CARD_RE.search(text):
        types.append("financial")
    if any(keyword in lowered for keyword in _PHI_KEYWORDS):
        types.append("PHI")
    if _SECRET_RE.search(text):
        types.append("operational")

    if not types:
        types.append("operational")

    # Keep a stable, unique order
    ordered = []
    for label in ("PII", "PHI", "financial", "operational"):
        if label in types and label not in ordered:
            ordered.append(label)
    return ordered


def _infer_sensitivity(text, data_types):
    """Map detected types (and explicit markers) to a sensitivity level."""
    lowered = (text or "").lower()
    if "PHI" in data_types:
        return "restricted"
    if "PII" in data_types or "financial" in data_types:
        return "confidential"
    if _SECRET_RE.search(text or ""):
        return "confidential"
    if any(marker in lowered for marker in _INTERNAL_MARKERS):
        return "internal"
    return "public"


def build_metadata(text, source):
    """
    Attach compliance metadata to a piece of text.

    Args:
        text:   The string being labeled.
        source: One of SOURCES — user_input, document, or model_output.

    Returns:
        A dict with sensitivity, data_type, and source. Values are primitives
        so they can be stored as ChromaDB metadata.
    """
    if source not in SOURCES:
        source = "document"
    data_types = _detected_types(text)
    return {
        "sensitivity": _infer_sensitivity(text, data_types),
        "data_type": ",".join(data_types),
        "source": source,
    }


def is_sensitive(metadata):
    """True when the record is not public operational content."""
    if not metadata:
        return False
    sensitivity = metadata.get("sensitivity", "public")
    data_type = metadata.get("data_type", "operational")
    return sensitivity != "public" or data_type != "operational"


def tag_documents(documents, source="document"):
    """Return a metadata dict for each document string."""
    return [build_metadata(doc, source) for doc in documents]


def redact_text(text):
    """
    Mask common sensitive patterns while leaving the rest of the text readable.

    Used as a safe default before logging, debug output, persistence, or
    sending text to the model.
    """
    if text is None:
        return ""
    redacted = str(text)
    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
    redacted = _PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    redacted = _SSN_RE.sub("[REDACTED_SSN]", redacted)
    redacted = _EMPLOYEE_ID_RE.sub("[REDACTED_ID]", redacted)
    redacted = _SECRET_RE.sub("[REDACTED_SECRET]", redacted)
    redacted = _CARD_RE.sub("[REDACTED_CARD]", redacted)
    return redacted


def prepare_for_model(text, metadata=None):
    """Redact text before it is sent to an LLM if it is tagged sensitive."""
    if metadata is None:
        metadata = build_metadata(text or "", "user_input")
    if is_sensitive(metadata):
        return redact_text(text)
    return text or ""


def safe_log(label, value):
    """Print a message with sensitive patterns stripped. Safe default for logs."""
    print(f"{label}: {redact_text(value)}")

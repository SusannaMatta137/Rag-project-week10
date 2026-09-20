# tests/test_basic.py
# -------------------
# Unit tests for deterministic, safety-critical helpers.
# These tests do not call Gemini or ChromaDB.

from compliance import (
    build_metadata,
    is_sensitive,
    prepare_for_model,
    redact_text,
    safe_log,
)
from filters import filter_by_threshold, has_relevant_results
from security import sanitize_input, validate_input


def test_redaction_removes_sensitive_text():
    input_text = "Contact me at test@example.com"
    output_text = redact_text(input_text)
    assert "@" not in output_text
    assert "[REDACTED_EMAIL]" in output_text


def test_redaction_masks_phone_ssn_and_employee_id():
    input_text = "Call 555-201-0147, SSN 123-45-6789, Employee ID EMP-4421"
    output_text = redact_text(input_text)
    assert "555-201-0147" not in output_text
    assert "123-45-6789" not in output_text
    assert "EMP-4421" not in output_text
    assert "[REDACTED_PHONE]" in output_text
    assert "[REDACTED_SSN]" in output_text
    assert "[REDACTED_ID]" in output_text


def test_metadata_tags_pii_as_confidential():
    metadata = build_metadata("Email jane.doe@example.com about onboarding", "user_input")
    assert metadata["source"] == "user_input"
    assert "PII" in metadata["data_type"]
    assert metadata["sensitivity"] == "confidential"
    assert is_sensitive(metadata)


def test_public_operational_text_is_not_sensitive():
    metadata = build_metadata("Python is a programming language.", "document")
    assert metadata["sensitivity"] == "public"
    assert metadata["data_type"] == "operational"
    assert metadata["source"] == "document"
    assert not is_sensitive(metadata)


def test_prepare_for_model_redacts_before_llm():
    query = "My email is student@school.edu"
    metadata = build_metadata(query, "user_input")
    prepared = prepare_for_model(query, metadata)
    assert "student@school.edu" not in prepared
    assert "@" not in prepared


def test_safe_log_does_not_print_raw_email(capsys):
    safe_log("User query", "Reach me at secret.user@example.com")
    captured = capsys.readouterr()
    assert "secret.user@example.com" not in captured.out
    assert "[REDACTED_EMAIL]" in captured.out


def test_sanitize_input_strips_whitespace():
    assert sanitize_input("  What is Python?  ") == "What is Python?"


def test_validate_input_blocks_empty_query():
    is_valid, message = validate_input("   ")
    assert is_valid is False
    assert "Please enter a question" in message


def test_filter_by_threshold_drops_distant_documents():
    docs, distances = filter_by_threshold(["keep", "drop"], [0.3, 1.5], threshold=1.0)
    assert docs == ["keep"]
    assert distances == [0.3]
    assert has_relevant_results(docs)
    assert not has_relevant_results([])

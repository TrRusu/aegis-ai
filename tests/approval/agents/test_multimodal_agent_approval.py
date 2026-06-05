"""
Approval tests for the multimodal agent — enrich_with_image().
"""
import json
from unittest.mock import MagicMock, patch

from approvaltests import verify

from agents.multimodal_agent import enrich_with_image


def test_enrich_with_image_no_image_returns_incident_unchanged():
    """Approval: enrich_with_image returns the original incident when image_bytes is None."""
    incident = "Ransomware detected on hospital network."

    result = enrich_with_image(incident, image_bytes=None)

    verify(result)


def test_enrich_with_image_empty_bytes_returns_incident_unchanged():
    """Approval: enrich_with_image returns the original incident when image_bytes is empty."""
    incident = "Suspicious login from unknown IP."

    result = enrich_with_image(incident, image_bytes=b"")

    verify(result)


@patch("agents.multimodal_agent.ChatOpenAI")
def test_enrich_with_image_with_image_return_type(mock_llm):
    """Approval: enrich_with_image returns a non-empty string when image bytes are provided."""
    mock_llm.return_value.invoke.return_value = MagicMock(
        content="Enriched: alert screenshot shows critical severity on host 192.168.1.5."
    )

    result = enrich_with_image(
        "Security alert observed.",
        image_bytes=b"fake-png-bytes",
        mime_type="image/png",
    )

    verify(json.dumps({
        "return_type": type(result).__name__,
        "is_non_empty": len(result) > 0,
    }, indent=2))


@patch("agents.multimodal_agent.ChatOpenAI")
def test_enrich_with_image_returns_llm_content(mock_llm):
    """Approval: enrich_with_image returns the enriched description from the LLM."""
    enriched = "Enriched: alert screenshot shows critical severity on host 192.168.1.5."
    mock_llm.return_value.invoke.return_value = MagicMock(content=enriched)

    result = enrich_with_image(
        "Security alert observed.",
        image_bytes=b"fake-png-bytes",
    )

    verify(result)


@patch("agents.multimodal_agent.ChatOpenAI")
def test_enrich_with_image_no_api_call_when_no_image(mock_llm):
    """Approval: no LLM call is made when image_bytes is None."""
    enrich_with_image("Incident description.", image_bytes=None)

    verify(json.dumps({"llm_called": mock_llm.called}, indent=2))

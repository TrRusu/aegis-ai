"""
Unit tests for rag/vision.py (TDD — written before the module exists).
"""
from unittest.mock import MagicMock, patch

import pytest


# ── _looks_like_chart_garbage ──────────────────────────────────────────────────

def test_looks_like_chart_garbage_returns_false_for_short_text():
    from rag.vision import _looks_like_chart_garbage
    assert _looks_like_chart_garbage("Hi") is False


def test_looks_like_chart_garbage_returns_false_for_real_prose():
    from rag.vision import _looks_like_chart_garbage
    text = "The average cost of a data breach increased significantly in 2024 according to IBM."
    assert _looks_like_chart_garbage(text) is False


def test_looks_like_chart_garbage_returns_true_for_ocr_noise():
    from rag.vision import _looks_like_chart_garbage
    text = "% US EU GB JP AU"
    assert _looks_like_chart_garbage(text) is True


def test_looks_like_chart_garbage_fewer_than_4_tokens_returns_false():
    from rag.vision import _looks_like_chart_garbage
    assert _looks_like_chart_garbage("A B C") is False


# ── _summarize_table ───────────────────────────────────────────────────────────

def test_summarize_table_calls_llm_with_table_text():
    from rag.vision import _summarize_table
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Summarized table.")

    result = _summarize_table("Col1 | Col2\n1 | 2", mock_llm)

    mock_llm.invoke.assert_called_once()
    assert result == "Summarized table."


def test_summarize_table_includes_doc_name_in_prompt():
    from rag.vision import _summarize_table
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Summary.")

    _summarize_table("some table", mock_llm, doc_name="IBM Report")

    call_args = mock_llm.invoke.call_args[0][0]
    assert any("IBM Report" in str(msg) for msg in call_args)


# ── _render_page_png ───────────────────────────────────────────────────────────

def test_render_page_png_returns_base64_string_on_success():
    from rag.vision import _render_page_png

    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"fake-png-bytes"

    mock_page = MagicMock()
    mock_page.get_pixmap.return_value = mock_pix

    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page

    with patch("rag.vision.fitz") as mock_fitz:
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()

        result = _render_page_png("report.pdf", 0)

    assert isinstance(result, str)
    assert len(result) > 0


def test_render_page_png_returns_none_on_failure():
    from rag.vision import _render_page_png

    with patch("rag.vision.fitz") as mock_fitz:
        mock_fitz.open.side_effect = Exception("file not found")

        result = _render_page_png("nonexistent.pdf", 0)

    assert result is None


# ── _summarize_page ────────────────────────────────────────────────────────────

def test_summarize_page_calls_llm_with_image_content():
    from rag.vision import _summarize_page
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Chart shows healthcare costs.")

    result = _summarize_page("base64imagedata", 3, mock_llm)

    mock_llm.invoke.assert_called_once()
    assert result == "Chart shows healthcare costs."


def test_summarize_page_includes_page_number_in_prompt():
    from rag.vision import _summarize_page
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Summary.")

    _summarize_page("base64data", 7, mock_llm, doc_name="IBM Report")

    call_args = str(mock_llm.invoke.call_args)
    assert "7" in call_args
    assert "IBM Report" in call_args

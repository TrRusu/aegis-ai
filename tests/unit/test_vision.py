"""
Unit tests for rag/vision.py
"""
from unittest.mock import MagicMock, patch

def test_looks_like_chart_garbage_returns_false_for_short_text():
    from rag.vision import VisionAnalyzer
    assert VisionAnalyzer.looks_like_chart_garbage("Hi") is False

def test_looks_like_chart_garbage_returns_false_for_real_prose():
    from rag.vision import VisionAnalyzer
    text = "The average cost of a data breach increased significantly in 2024 according to IBM."
    assert VisionAnalyzer.looks_like_chart_garbage(text) is False

def test_looks_like_chart_garbage_returns_true_for_ocr_noise():
    from rag.vision import VisionAnalyzer
    assert VisionAnalyzer.looks_like_chart_garbage("% US EU GB JP AU") is True

def test_looks_like_chart_garbage_fewer_than_4_tokens_returns_false():
    from rag.vision import VisionAnalyzer
    assert VisionAnalyzer.looks_like_chart_garbage("A B C") is False

def test_summarize_table_calls_llm_and_returns_content():
    from rag.vision import VisionAnalyzer
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Summarized table.")

    analyzer = VisionAnalyzer(llm=mock_llm)
    result = analyzer.summarize_table("Col1 | Col2\n1 | 2")

    mock_llm.invoke.assert_called_once()
    assert result == "Summarized table."

def test_summarize_table_includes_doc_name_in_prompt():
    from rag.vision import VisionAnalyzer
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Summary.")

    analyzer = VisionAnalyzer(llm=mock_llm)
    analyzer.summarize_table("some table", doc_name="IBM Report")

    call_args = str(mock_llm.invoke.call_args)
    assert "IBM Report" in call_args

def test_render_page_png_returns_base64_string_on_success():
    from rag.vision import VisionAnalyzer

    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"fake-png-bytes"
    mock_page = MagicMock()
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page

    with patch("rag.vision.fitz") as mock_fitz:
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()
        result = VisionAnalyzer.render_page_png("report.pdf", 0)

    assert isinstance(result, str)
    assert len(result) > 0

def test_render_page_png_returns_none_on_failure():
    from rag.vision import VisionAnalyzer

    with patch("rag.vision.fitz") as mock_fitz:
        mock_fitz.open.side_effect = Exception("file not found")
        result = VisionAnalyzer.render_page_png("nonexistent.pdf", 0)

    assert result is None

def test_summarize_page_calls_llm_and_returns_content():
    from rag.vision import VisionAnalyzer
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Chart shows healthcare costs.")

    analyzer = VisionAnalyzer(llm=mock_llm)
    result = analyzer.summarize_page("base64imagedata", 3)

    mock_llm.invoke.assert_called_once()
    assert result == "Chart shows healthcare costs."

def test_summarize_page_includes_page_number_and_doc_name_in_prompt():
    from rag.vision import VisionAnalyzer
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Summary.")

    analyzer = VisionAnalyzer(llm=mock_llm)
    analyzer.summarize_page("base64data", 7, doc_name="IBM Report")

    call_args = str(mock_llm.invoke.call_args)
    assert "7" in call_args
    assert "IBM Report" in call_args
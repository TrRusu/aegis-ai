"""
Approval tests for the CVE MCP server
"""
from unittest.mock import patch

from approvaltests import verify

from mcp_servers.cve_server import lookup_cve


def _make_nvd_response(
    cve_id="CVE-2021-44228",
    published="2021-12-10T10:15:00.000",
    status="Analyzed",
    description="A critical RCE vulnerability in Log4j.",
    cvss_key="cvssMetricV31",
    base_score=10.0,
    base_severity="CRITICAL",
    version="3.1",
):
    return {
        "vulnerabilities": [
            {
                "cve": {
                    "id": cve_id,
                    "published": published,
                    "vulnStatus": status,
                    "descriptions": [{"lang": "en", "value": description}],
                    "metrics": {
                        cvss_key: [
                            {
                                "cvssData": {"baseScore": base_score, "version": version},
                                "baseSeverity": base_severity,
                            }
                        ]
                    },
                }
            }
        ]
    }

@patch("mcp_servers.cve_server.httpx.get")
def test_lookup_cve_returns_formatted_result(mock_get):
    """Approval: lookup_cve returns a formatted string with all CVE fields."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = _make_nvd_response()

    result = lookup_cve("CVE-2021-44228")

    verify(result)

@patch("mcp_servers.cve_server.httpx.get")
def test_lookup_cve_falls_back_to_cvss_v30(mock_get):
    """Approval: lookup_cve uses CVSS v3.0 when v3.1 is not present."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = _make_nvd_response(
        cvss_key="cvssMetricV30",
        base_score=8.8,
        base_severity="HIGH",
        version="3.0",
    )

    result = lookup_cve("CVE-2021-44228")

    verify(result)

@patch("mcp_servers.cve_server.httpx.get")
def test_lookup_cve_falls_back_to_cvss_v2(mock_get):
    """Approval: lookup_cve uses CVSS v2 when neither v3.1 nor v3.0 are present."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = _make_nvd_response(
        cvss_key="cvssMetricV2",
        base_score=7.8,
        base_severity="HIGH",
        version="2.0",
    )

    result = lookup_cve("CVE-2021-44228")

    verify(result)

@patch("mcp_servers.cve_server.httpx.get")
def test_lookup_cve_adds_prefix_when_missing(mock_get):
    """Approval: lookup_cve prepends 'CVE-' when the input omits it."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = _make_nvd_response(cve_id="CVE-2021-44228")

    lookup_cve("2021-44228")

    called_with = mock_get.call_args[1]["params"]["cveId"]
    verify(called_with)

@patch("mcp_servers.cve_server.httpx.get")
def test_lookup_cve_not_found_in_nvd(mock_get):
    """Approval: lookup_cve returns a not-found message when NVD has no record."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"vulnerabilities": []}

    result = lookup_cve("CVE-9999-99999")

    verify(result)

@patch("mcp_servers.cve_server.httpx.get")
def test_lookup_cve_non_200_status(mock_get):
    """Approval: lookup_cve returns an error message on non-200 HTTP status."""
    mock_get.return_value.status_code = 404

    result = lookup_cve("CVE-2021-44228")

    verify(result)

@patch("mcp_servers.cve_server.httpx.get")
def test_lookup_cve_timeout(mock_get):
    """Approval: lookup_cve returns a timeout message when the NVD API times out."""
    import httpx
    mock_get.side_effect = httpx.TimeoutException("timed out")

    result = lookup_cve("CVE-2021-44228")

    verify(result)

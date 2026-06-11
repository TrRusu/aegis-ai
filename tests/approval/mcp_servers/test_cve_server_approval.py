"""
Approval tests for the CVE MCP server
"""
from unittest.mock import MagicMock, patch

from approvaltests import verify

from mcp_servers.cve_server import lookup_cve


def _mock_lookup(return_value: str):
    mock = MagicMock()
    mock.lookup.return_value = return_value
    return mock

@patch("mcp_servers.cve_server._cve_lookup")
def test_lookup_cve_returns_formatted_result(mock_cve_lookup):
    """Approval: lookup_cve returns a formatted string with all CVE fields."""
    mock_cve_lookup.lookup.return_value = (
        "CVE ID:      CVE-2021-44228\n"
        "Published:   2021-12-10\n"
        "Status:      Analyzed\n"
        "CVSS 3.1: 10.0 (CRITICAL)\n"
        "Description: A critical RCE vulnerability in Log4j."
    )
    verify(lookup_cve("CVE-2021-44228"))

@patch("mcp_servers.cve_server._cve_lookup")
def test_lookup_cve_falls_back_to_cvss_v30(mock_cve_lookup):
    """Approval: lookup_cve uses CVSS v3.0 when v3.1 is not present."""
    mock_cve_lookup.lookup.return_value = (
        "CVE ID:      CVE-2021-44228\n"
        "Published:   2021-12-10\n"
        "Status:      Analyzed\n"
        "CVSS 3.0: 8.8 (HIGH)\n"
        "Description: A critical RCE vulnerability in Log4j."
    )
    verify(lookup_cve("CVE-2021-44228"))

@patch("mcp_servers.cve_server._cve_lookup")
def test_lookup_cve_falls_back_to_cvss_v2(mock_cve_lookup):
    """Approval: lookup_cve uses CVSS v2 when neither v3.1 nor v3.0 are present."""
    mock_cve_lookup.lookup.return_value = (
        "CVE ID:      CVE-2021-44228\n"
        "Published:   2021-12-10\n"
        "Status:      Analyzed\n"
        "CVSS 2.0: 7.8 (HIGH)\n"
        "Description: A critical RCE vulnerability in Log4j."
    )
    verify(lookup_cve("CVE-2021-44228"))

@patch("mcp_servers.cve_lookup.httpx.Client")
def test_lookup_cve_adds_prefix_when_missing(mock_client_cls):
    """Approval: lookup_cve prepends 'CVE-' and passes the normalised ID to NVD."""
    from mcp_servers.cve_server import _cve_lookup
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"vulnerabilities": []}
    mock_client_cls.return_value.get.return_value = mock_resp
    _cve_lookup._client = mock_client_cls.return_value
    lookup_cve("2021-44228")
    called_with = mock_client_cls.return_value.get.call_args[1]["params"]["cveId"]
    verify(called_with)

@patch("mcp_servers.cve_server._cve_lookup")
def test_lookup_cve_not_found_in_nvd(mock_cve_lookup):
    """Approval: lookup_cve returns a not-found message when NVD has no record."""
    mock_cve_lookup.lookup.return_value = "No record found in NVD for CVE-9999-99999."
    verify(lookup_cve("CVE-9999-99999"))

@patch("mcp_servers.cve_server._cve_lookup")
def test_lookup_cve_non_200_status(mock_cve_lookup):
    """Approval: lookup_cve returns an error message on non-200 HTTP status."""
    mock_cve_lookup.lookup.return_value = "NVD API returned status 404 for CVE-2021-44228."
    verify(lookup_cve("CVE-2021-44228"))

@patch("mcp_servers.cve_server._cve_lookup")
def test_lookup_cve_timeout(mock_cve_lookup):
    """Approval: lookup_cve returns a timeout message when the NVD API times out."""
    mock_cve_lookup.lookup.return_value = "NVD API timed out looking up CVE-2021-44228."
    verify(lookup_cve("CVE-2021-44228"))

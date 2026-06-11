"""
Unit tests for CVELookup class in mcp_servers/cve_server.py (TDD).
"""
from unittest.mock import MagicMock


def _make_nvd_response(cve_id="CVE-2021-44228", score=10.0, severity="CRITICAL", version="3.1"):
    return {
        "vulnerabilities": [{
            "cve": {
                "id": cve_id,
                "published": "2021-12-10T00:00:00.000",
                "vulnStatus": "Analyzed",
                "descriptions": [{"lang": "en", "value": "Log4Shell RCE vulnerability."}],
                "metrics": {
                    f"cvssMetricV{version.replace('.', '')}": [{
                        "baseSeverity": severity,
                        "cvssData": {"baseScore": score, "version": version},
                    }]
                },
            }
        }]
    }


def test_cve_lookup_normalises_id_without_prefix():
    from mcp_servers.cve_lookup import CVELookup
    mock_client = MagicMock()
    mock_client.get.return_value.status_code = 200
    mock_client.get.return_value.json.return_value = _make_nvd_response("CVE-2021-44228")
    result = CVELookup(client=mock_client).lookup("2021-44228")
    call_params = mock_client.get.call_args
    assert call_params.kwargs["params"]["cveId"] == "CVE-2021-44228"


def test_cve_lookup_keeps_cve_prefix_if_already_present():
    from mcp_servers.cve_lookup import CVELookup
    mock_client = MagicMock()
    mock_client.get.return_value.status_code = 200
    mock_client.get.return_value.json.return_value = _make_nvd_response()
    CVELookup(client=mock_client).lookup("CVE-2021-44228")
    call_params = mock_client.get.call_args
    assert call_params.kwargs["params"]["cveId"] == "CVE-2021-44228"


def test_cve_lookup_returns_formatted_string_on_success():
    from mcp_servers.cve_lookup import CVELookup
    mock_client = MagicMock()
    mock_client.get.return_value.status_code = 200
    mock_client.get.return_value.json.return_value = _make_nvd_response()
    result = CVELookup(client=mock_client).lookup("CVE-2021-44228")
    assert "CVE-2021-44228" in result
    assert "CRITICAL" in result
    assert "10.0" in result


def test_cve_lookup_returns_error_on_non_200_status():
    from mcp_servers.cve_lookup import CVELookup
    mock_client = MagicMock()
    mock_client.get.return_value.status_code = 404
    result = CVELookup(client=mock_client).lookup("CVE-2021-44228")
    assert "404" in result


def test_cve_lookup_returns_error_when_no_vulnerabilities():
    from mcp_servers.cve_lookup import CVELookup
    mock_client = MagicMock()
    mock_client.get.return_value.status_code = 200
    mock_client.get.return_value.json.return_value = {"vulnerabilities": []}
    result = CVELookup(client=mock_client).lookup("CVE-2021-44228")
    assert "No record found" in result


def test_cve_lookup_returns_error_on_timeout():
    from mcp_servers.cve_lookup import CVELookup
    import httpx
    mock_client = MagicMock()
    mock_client.get.side_effect = httpx.TimeoutException("timed out")
    result = CVELookup(client=mock_client).lookup("CVE-2021-44228")
    assert "timed out" in result.lower()


def test_cve_lookup_uses_injected_client():
    from mcp_servers.cve_lookup import CVELookup
    mock_client = MagicMock()
    mock_client.get.return_value.status_code = 200
    mock_client.get.return_value.json.return_value = _make_nvd_response()
    CVELookup(client=mock_client).lookup("CVE-2021-44228")
    mock_client.get.assert_called_once()

import pytest
import httpx
from mcp_servers.cve_lookup import CVELookup


@pytest.mark.integration
def test_lookup_known_cve_returns_id():
    assert "CVE-2021-44228" in CVELookup().lookup("CVE-2021-44228")


@pytest.mark.integration
def test_lookup_known_cve_returns_cvss_score():
    result = CVELookup().lookup("CVE-2021-44228")
    assert "CVSS" in result
    assert "10.0" in result


@pytest.mark.integration
def test_lookup_known_cve_returns_description():
    result = CVELookup().lookup("CVE-2021-44228")
    assert "Description:" in result
    assert len(result) > 100


@pytest.mark.integration
def test_lookup_normalises_id_without_prefix():
    assert "CVE-2021-44228" in CVELookup().lookup("2021-44228")


@pytest.mark.integration
def test_lookup_nonexistent_cve_returns_not_found():
    assert "No record found" in CVELookup().lookup("CVE-9999-99999")


@pytest.mark.integration
def test_lookup_nvd_timeout_returns_message(monkeypatch):
    client = httpx.Client()
    monkeypatch.setattr(client, "get", lambda *a, **kw: (_ for _ in ()).throw(httpx.TimeoutException("timeout")))
    assert "timed out" in CVELookup(client=client).lookup("CVE-2021-44228")

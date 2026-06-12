import pytest
from mcp_servers.cve_lookup import CVELookup


@pytest.mark.integration
class TestCVELookupIntegration:

    def test_lookup_known_cve_returns_id(self):
        result = CVELookup().lookup("CVE-2021-44228")
        assert "CVE-2021-44228" in result

    def test_lookup_known_cve_returns_cvss_score(self):
        result = CVELookup().lookup("CVE-2021-44228")
        assert "CVSS" in result
        assert "10.0" in result

    def test_lookup_known_cve_returns_description(self):
        result = CVELookup().lookup("CVE-2021-44228")
        assert "Description:" in result
        assert len(result) > 100

    def test_lookup_normalises_id_without_prefix(self):
        result = CVELookup().lookup("2021-44228")
        assert "CVE-2021-44228" in result

    def test_lookup_nonexistent_cve_returns_not_found(self):
        result = CVELookup().lookup("CVE-9999-99999")
        assert "No record found" in result

    def test_lookup_nvd_timeout_returns_message(self, monkeypatch):
        import httpx

        def raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("timeout")

        client = httpx.Client()
        monkeypatch.setattr(client, "get", raise_timeout)
        result = CVELookup(client=client).lookup("CVE-2021-44228")
        assert "timed out" in result

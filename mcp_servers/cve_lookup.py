"""Looks up CVE records from the National Vulnerability Database with an injectable HTTP client.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx

NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"


class CVELookup:

    def __init__(self, client: httpx.Client = None):
        self._client = client or httpx.Client()

    def lookup(self, cve_id: str) -> str:
        cve_id = cve_id.strip().upper()
        if not cve_id.startswith("CVE-"):
            cve_id = f"CVE-{cve_id}"

        try:
            resp = self._client.get(NVD_API, params={"cveId": cve_id}, timeout=10)
        except httpx.TimeoutException:
            return f"NVD API timed out looking up {cve_id}."

        if resp.status_code != 200:
            return f"NVD API returned status {resp.status_code} for {cve_id}."

        vulns = resp.json().get("vulnerabilities", [])
        if not vulns:
            return f"No record found in NVD for {cve_id}."

        cve = vulns[0]["cve"]
        description = next(
            (d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"),
            "No English description available.",
        )

        metrics = cve.get("metrics", {})
        cvss_score = "N/A"
        severity = "N/A"
        cvss_version = "N/A"
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in metrics and metrics[key]:
                m = metrics[key][0]
                data = m.get("cvssData", {})
                cvss_score = data.get("baseScore", "N/A")
                severity = m.get("baseSeverity", data.get("baseSeverity", "N/A"))
                cvss_version = data.get("version", key)
                break

        published = cve.get("published", "Unknown")[:10]
        status = cve.get("vulnStatus", "Unknown")

        return (
            f"CVE ID:      {cve['id']}\n"
            f"Published:   {published}\n"
            f"Status:      {status}\n"
            f"CVSS {cvss_version}: {cvss_score} ({severity})\n"
            f"Description: {description}"
        )

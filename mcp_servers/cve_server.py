"""CVE Lookup MCP Server"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.server.fastmcp import FastMCP
from mcp_servers.cve_lookup import CVELookup

mcp = FastMCP("CVE Lookup")
_cve_lookup = CVELookup()

@mcp.tool()
def lookup_cve(cve_id: str) -> str:
    """Look up a CVE by ID from the National Vulnerability Database (NVD).
    Returns the severity score, CVSS rating, publish date, status and description.
    Use this when the user asks about a specific vulnerability or CVE ID."""
    return _cve_lookup.lookup(cve_id)


if __name__ == "__main__":
    mcp.run()

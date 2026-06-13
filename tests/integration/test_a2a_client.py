import pytest
from agents.a2a_client import A2AClient
from app.config import A2A_SERVER_URL


@pytest.fixture(autouse=True)
def require_a2a_server():
    if not A2AClient(base_url=A2A_SERVER_URL).is_server_available():
        pytest.skip("A2A server not running — start it with: python a2a_server/threat_intel_server.py")


@pytest.fixture
def client():
    return A2AClient(base_url=A2A_SERVER_URL)


@pytest.mark.integration
def test_server_is_available(client):
    assert client.is_server_available() is True


@pytest.mark.integration
def test_fetch_agent_card_returns_name(client):
    card = client.fetch_agent_card()
    assert "name" in card
    assert isinstance(card["name"], str)


@pytest.mark.integration
def test_fetch_agent_card_returns_skills(client):
    card = client.fetch_agent_card()
    assert "skills" in card
    assert len(card["skills"]) > 0


@pytest.mark.integration
def test_call_agent_returns_analysis(client):
    analysis = client.call_threat_intel_agent(
        "Unauthorized access detected on production server. Possible CVE-2021-44228 exploit."
    )
    assert isinstance(analysis, str)
    assert len(analysis) > 50

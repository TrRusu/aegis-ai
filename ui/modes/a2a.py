import streamlit as st
from agents.a2a_client import A2AClient


def handle(prompt: str, a2a_client: A2AClient) -> str:
    if not a2a_client.is_server_available():
        response = "Remote Threat Intelligence Agent is offline. Start it with: `python a2a_server/threat_intel_server.py`"
        st.error(response)
        return response
    with st.spinner("Calling remote Threat Intelligence Agent at localhost:8888..."):
        try:
            response = a2a_client.call_threat_intel_agent(prompt)
            st.markdown(response)
            st.caption("Analysis provided by remote ThreatIntelAgent via A2A protocol.")
        except Exception as exc:
            response = f"A2A call failed: {exc}"
            st.error(response)
    return response

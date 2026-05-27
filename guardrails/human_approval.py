"""
Human-in-the-Loop approval store.

Tracks pending approval proposals in Streamlit session state.
Each proposal has: incident, preliminary_analysis, severity, tool_calls_log.
"""

APPROVAL_SEVERITIES = {"critical", "high"}


def requires_approval(severity: str) -> bool:
    return severity.strip().lower() in APPROVAL_SEVERITIES


def save_pending(session_state, incident: str, severity: str,
                 preliminary_analysis: str, tool_calls_log: list) -> None:
    session_state["pending_approval"] = {
        "incident": incident,
        "severity": severity,
        "preliminary_analysis": preliminary_analysis,
        "tool_calls_log": tool_calls_log,
    }


def get_pending(session_state) -> dict | None:
    return session_state.get("pending_approval")


def clear_pending(session_state) -> None:
    session_state.pop("pending_approval", None)

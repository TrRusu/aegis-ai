import streamlit as st
from agents.supervisor_hitl import HitlSupervisor, requires_approval
from ui.shared import build_llm
from ui.components.chat import render_response
from ui.components.tool_calls import render_tool_calls


def render_approval_card() -> None:
    if "hitl_pending" not in st.session_state:
        return
    pending = st.session_state["hitl_pending"]
    st.divider()
    st.subheader("Pending Approval")
    st.caption(f"Severity: **{pending['severity']}** — review the Phase 1 briefing above then decide.")
    reason = st.text_input("Your reason (optional)", key="hitl_reason", placeholder="e.g. Confirmed with SOC team")
    col1, col2 = st.columns(2)
    if col1.button("Approve — proceed with full response", use_container_width=True, type="primary"):
        with st.spinner("Phase 2 — compiling approved response plan..."):
            _llm = build_llm(temperature=0.2, max_tokens=2048)
            final = HitlSupervisor(llm=_llm).run_phase2(pending["incident"], pending["preliminary"], "APPROVED", reason or "Approved by analyst.")
        st.session_state.messages.append({"role": "assistant", "content": final})
        del st.session_state["hitl_pending"]
        st.rerun()
    if col2.button("Reject — use conservative fallback", use_container_width=True):
        with st.spinner("Phase 2 — compiling rejection response..."):
            _llm = build_llm(temperature=0.2, max_tokens=2048)
            final = HitlSupervisor(llm=_llm).run_phase2(pending["incident"], pending["preliminary"], "REJECTED", reason or "Rejected by analyst.")
        st.session_state.messages.append({"role": "assistant", "content": final})
        del st.session_state["hitl_pending"]
        st.rerun()
    st.divider()


def handle(prompt: str, temperature: float, max_tokens: int, hitl_k: int) -> str:
    with st.spinner("Phase 1 — gathering intelligence..."):
        _llm = build_llm(temperature=temperature, max_tokens=max_tokens)
        preliminary, severity, tool_calls_log = HitlSupervisor(llm=_llm).run_phase1(prompt, k=hitl_k)

    if requires_approval(severity):
        response = f"**Phase 1 complete — Severity: {severity}**\n\nThis incident requires human approval before proceeding.\n\n---\n\n{preliminary}"
        st.warning(f"Severity **{severity}** — human approval required before final report.")
        render_response(preliminary)
        st.session_state["hitl_pending"] = {
            "incident": prompt,
            "preliminary": preliminary,
            "tool_calls_log": tool_calls_log,
            "severity": severity,
        }
        st.session_state["hitl_needs_rerun"] = True
    else:
        with st.spinner("Phase 2 — compiling final report (no approval needed)..."):
            _llm = build_llm(temperature=temperature, max_tokens=max_tokens)
            final = HitlSupervisor(llm=_llm).run_phase2(prompt, preliminary, "AUTO-APPROVED", f"Severity {severity} is below approval threshold.")
        response = final
        render_response(final)

    render_tool_calls(tool_calls_log, label="Phase 1 specialist agents")
    return response

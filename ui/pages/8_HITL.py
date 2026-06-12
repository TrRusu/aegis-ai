import streamlit as st
from agents.supervisor_hitl import HitlSupervisor, requires_approval
from ui.components.chat import render_chat_history, append_message, render_response
from ui.components.sidebar import render_model_params, render_k_slider, render_sidebar_footer
from ui.components.tool_calls import render_tool_calls
from ui.shared import check_injection, build_llm

st.set_page_config(page_title="HITL — Aegis", page_icon="🛡️", layout="centered")
st.title("🛡️ HITL Supervisor")

with st.sidebar:
    temperature, max_tokens = render_model_params()
    st.divider()
    st.header("HITL Supervisor Settings")
    hitl_k = render_k_slider(default=6)
    st.divider()
    st.info("Critical/High severity incidents require human approval before the final report is produced. The workflow pauses after Phase 1 and resumes only after you approve or reject.")
    render_sidebar_footer()

if "messages" not in st.session_state:
    st.session_state.messages = []

render_chat_history()

if "hitl_pending" in st.session_state:
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
        append_message("assistant", final)
        del st.session_state["hitl_pending"]
        st.rerun()
    if col2.button("Reject — use conservative fallback", use_container_width=True):
        with st.spinner("Phase 2 — compiling rejection response..."):
            _llm = build_llm(temperature=0.2, max_tokens=2048)
            final = HitlSupervisor(llm=_llm).run_phase2(pending["incident"], pending["preliminary"], "REJECTED", reason or "Rejected by analyst.")
        append_message("assistant", final)
        del st.session_state["hitl_pending"]
        st.rerun()
    st.divider()

if prompt := st.chat_input("Ask Aegis about a threat, CVE, or incident..."):
    append_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        is_injection, injection_score = check_injection(prompt)

        if is_injection:
            response = f"Message blocked — potential prompt injection detected (risk score: {injection_score:.2f}). Please rephrase your question."
            st.warning(response)
        else:
            with st.spinner("Phase 1 — gathering intelligence..."):
                _llm = build_llm(temperature=temperature, max_tokens=max_tokens)
                preliminary, severity, tool_calls_log = HitlSupervisor(llm=_llm).run_phase1(prompt, k=hitl_k)

            if requires_approval(severity):
                response = f"**Phase 1 complete — Severity: {severity}**\n\nThis incident requires human approval before proceeding.\n\n---\n\n{preliminary}"
                st.warning(f"Severity **{severity}** — human approval required before final report.")
                st.markdown(preliminary.replace("$", r"\$"))
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
                st.markdown(final.replace("$", r"\$"))

            render_tool_calls(tool_calls_log, label="Phase 1 specialist agents")

    append_message("assistant", response)
    if st.session_state.pop("hitl_needs_rerun", False):
        st.rerun()

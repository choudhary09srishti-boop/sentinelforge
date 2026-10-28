"""SentinelForge Streamlit prototype. Talks only to the HTTP API."""

import time

import httpx
import streamlit as st

st.set_page_config(page_title="SentinelForge", layout="wide")
st.sidebar.title("SentinelForge")
BASE = st.sidebar.text_input("API URL", "http://127.0.0.1:8000").rstrip("/")
KEY = st.sidebar.text_input("API key", "change-me", type="password")


def call(method: str, path: str, **kwargs):
    try:
        resp = httpx.request(
            method, BASE + path, headers={"X-API-Key": KEY}, timeout=130, **kwargs
        )
    except httpx.HTTPError:
        st.error("Cannot reach the API. Is the backend running?")
        return None
    if resp.status_code >= 400:
        try:
            msg = resp.json()["error"]["message"]
        except Exception:
            msg = resp.text[:200]
        st.error(f"{resp.status_code}: {msg}")
        return None
    return resp.json()


def show_hits(hits: list) -> None:
    for i, h in enumerate(hits, 1):
        c = h["citation"]
        page = f" p.{c['page']}" if c.get("page") else ""
        title = f"[{i}] {c['source']}{page} lines {c['start_line']}-{c['end_line']}"
        with st.expander(f"{title}  (score {h['score']:.3f})"):
            if h["risk_flags"]:
                st.warning("Possible prompt injection: " + ", ".join(h["risk_flags"]))
            st.text(h["text"])


tab_docs, tab_chat, tab_agent, tab_health = st.tabs(
    ["Documents", "Chat", "Agent", "Health"]
)

with tab_docs:
    up = st.file_uploader("Upload a document")
    if up and st.button("Index document"):
        res = call("POST", "/api/v1/documents", files={"file": (up.name, up.getvalue())})
        if res:
            st.success(
                f"{res['name']}: {res['status']} ({res['chunks_total']} chunks, "
                f"{res['risky_chunks']} flagged risky)"
            )
    if st.button("Index workspace"):
        res = call(
            "POST", "/api/v1/documents/index-workspace", json={"path": ".", "max_files": 50}
        )
        if res:
            st.json(res)
    data = call("GET", "/api/v1/documents")
    if data:
        st.caption(f"{len(data['documents'])} documents, {data['total_chunks']} chunks")
        for d in data["documents"]:
            c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
            c1.write(d["name"])
            c2.write(f"{d['chunk_count']} chunks")
            c3.write(f"{d['risky_chunks']} risky")
            if c4.button("Delete", key="del" + d["document_id"]):
                call("DELETE", "/api/v1/documents/" + d["document_id"])
                st.rerun()

with tab_chat:
    mode = st.selectbox("Search mode", ["hybrid", "semantic", "keyword"])
    history = st.session_state.setdefault("chat", [])
    question = st.chat_input("Ask about your documents")
    if question:
        history.append({"role": "user", "text": question, "hits": []})
        res = call("POST", "/api/v1/search", json={"query": question, "top_k": 5, "mode": mode})
        if res:
            text = (
                f"{len(res['hits'])} passages found ({res['mode_used']})."
                if res["hits"]
                else "No matching passages."
            )
            history.append({"role": "assistant", "text": text, "hits": res["hits"]})
    for m in history:
        with st.chat_message(m["role"]):
            st.write(m["text"])
            show_hits(m["hits"])

with tab_agent:
    task = st.text_area("Task for the coding agent", height=80)
    if st.button("Start run", type="primary", disabled=len(task.strip()) < 5):
        run = call("POST", "/api/v1/agent/runs", json={"task": task.strip()})
        if run:
            st.session_state["run_id"] = run["id"]
    runs = call("GET", "/api/v1/agent/runs")
    runs_list = runs["runs"] if runs else []
    ids = [r["id"] for r in runs_list]
    if ids:
        labels = {r["id"]: f"{r['status']} | {r['task'][:60]}" for r in runs_list}
        current = st.session_state.get("run_id")
        index = ids.index(current) if current in ids else 0
        run_id = st.selectbox("Run", ids, index=index, format_func=labels.get)
        st.session_state["run_id"] = run_id
        st.button("Refresh")
        base_path = f"/api/v1/agent/runs/{run_id}"
        d = call("GET", base_path)
        deadline = time.time() + 180
        while d and d["status"] in ("queued", "running") and time.time() < deadline:
            with st.spinner("Agent is working..."):
                time.sleep(1.5)
            d = call("GET", base_path)
        if d:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Status", d["status"])
            m2.metric("Tests", d["test_status"])
            m3.metric("Attempts", d["attempts"])
            m4.metric("Review OK", str(d["review_ok"]))
            if d["error"]:
                st.error(d["error"])
            if d["plan"]:
                st.markdown("**Plan**\n" + "\n".join(f"{n}. {s}" for n, s in enumerate(d["plan"], 1)))
            if d["diff"]:
                st.markdown("**Diff preview**")
                st.code(d["diff"], language="diff")
            for risk in d["risks"]:
                st.warning(risk)
            if d["test_output"]:
                with st.expander("Test output"):
                    st.code(d["test_output"])
            if d["evidence"]:
                with st.expander("Evidence"):
                    st.write("\n\n".join(d["evidence"]))
            if d["report"]:
                with st.expander("Report"):
                    st.markdown(d["report"])
            if d["apply_result"]:
                st.success(f"Applied to {d['apply_result']['path']}")
            if d["reject_reason"]:
                st.info(f"Rejected: {d['reject_reason']}")
            if d["status"] == "awaiting_approval":
                reason = st.text_input("Reject reason (optional)")
                a, r = st.columns(2)
                if a.button("Approve and apply", type="primary"):
                    if call("POST", base_path + "/approve"):
                        st.rerun()
                if r.button("Reject"):
                    if call("POST", base_path + "/reject", json={"reason": reason}):
                        st.rerun()
            logs = call("GET", base_path + "/logs?after=0&limit=500")
            with st.expander("Logs", expanded=True):
                if logs:
                    st.code(
                        "\n".join(
                            f"{e['ts'][11:19]} {e['level']:<7} {e['message']}"
                            for e in logs["events"]
                        )
                        or "No log events yet."
                    )

with tab_health:
    st.json(call("GET", "/health") or {})
    st.json(call("GET", "/api/v1/config") or {})

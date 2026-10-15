"""Streamlit UI: durable HITL review and all four bonus challenges."""
from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from audit.bonuses import calibration, checkpoint_history, fork_checkpoint, recent_sessions
from common.db import db_path, replay_events
from common.github import parse_pr_url
from exercises.exercise_4_audit import build_graph

load_dotenv()
st.set_page_config(page_title="HITL PR Review", layout="wide")


async def run_graph(pr_url, thread_id, resume_value=None, *, restore=False,
                    reviewers=None, dry_run=False):
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if restore:
            snapshot = await app.aget_state(cfg)
            if not snapshot.values:
                raise ValueError("No saved checkpoint found for this session")
            result = dict(snapshot.values)
            if snapshot.interrupts:
                result["__interrupt__"] = snapshot.interrupts
            elif snapshot.next:
                result = await app.ainvoke(None, cfg)
            return result
        if resume_value is not None:
            return await app.ainvoke(Command(resume=resume_value), cfg)
        return await app.ainvoke(dict(pr_url=pr_url, thread_id=thread_id,
                                     reviewers=reviewers or [], dry_run=dry_run), cfg)


async def history_for(thread_id):
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        return await checkpoint_history(build_graph(cp), thread_id)


async def fork_for(thread_id, checkpoint_id):
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        return await fork_checkpoint(build_graph(cp), thread_id, checkpoint_id)


def accept_result(result):
    st.session_state.pr_url = result.get("pr_url", st.session_state.pr_url)
    st.session_state.pending = list(result.get("__interrupt__", ()))
    st.session_state.final = None if st.session_state.pending else result
    st.session_state.current = result


def render_approval_card(payload, key="approval"):
    st.subheader(f"Approval requested — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")
    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")
    feedback = st.text_area("Feedback (required for Edit)", key=key + "_feedback")
    cols = st.columns(3)
    for col, choice in zip(cols, ("approve", "reject", "edit")):
        if col.button(choice.title(), key=key + "_" + choice,
                      type="primary" if choice == "approve" else "secondary"):
            if choice == "edit" and not feedback.strip():
                st.error("Describe the requested edit first.")
                return None
            return {"choice": choice, "feedback": feedback.strip()}
    return None


def render_escalation_card(payload, key="escalation"):
    label = f" — reviewer {payload['reviewer']}" if payload.get("reviewer") else ""
    st.subheader(f"Strong escalation{label} — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + "; ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])
    if payload.get("reviewer_thread_id"):
        st.caption("Reviewer thread: " + payload["reviewer_thread_id"])
    with st.form(key):
        answers = {q: st.text_input(q, key=f"{key}_q_{i}")
                   for i, q in enumerate(payload["questions"])}
        submitted = st.form_submit_button("Submit answers")
    if submitted:
        if any(not a.strip() for a in answers.values()):
            st.error("Please answer every question.")
            return None
        return {q: a.strip() for q, a in answers.items()}
    return None


def main():
    defaults = dict(thread_id=None, pr_url="", pending=[], final=None, current={})
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # URL stores only the session ID, allowing a browser refresh to restore SQLite state.
    url_thread = st.query_params.get("thread")
    if url_thread and url_thread != st.session_state.thread_id:
        st.session_state.thread_id = url_thread
        try:
            accept_result(asyncio.run(run_graph("", url_thread, restore=True)))
        except Exception as exc:
            st.error(f"Could not restore session: {exc}")
            st.query_params.clear()
            st.session_state.thread_id = None
    st.title("HITL PR Review Agent")

    with st.sidebar:
        st.header("Recent sessions")
        sessions = asyncio.run(recent_sessions())
        if sessions:
            event = st.dataframe(sessions, hide_index=True, on_select="rerun",
                                 selection_mode="single-row", key="sessions_" + (st.session_state.thread_id or 'none'))
            selected = event.selection.rows
            if selected:
                row = sessions[selected[0]]
                if row["thread_id"] != st.session_state.thread_id:
                    st.query_params["thread"] = row["thread_id"]
                    st.session_state.thread_id = row["thread_id"]
                    accept_result(asyncio.run(run_graph(row["pr_url"], row["thread_id"], restore=True)))
                    st.rerun()
        else:
            st.caption("No review sessions yet.")

    with st.form("start"):
        pr_url = st.text_input("PR URL", value=st.session_state.pr_url,
                              placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1")
        dry_run = st.checkbox("Dry run — do not post a GitHub comment", value=True)
        multi = st.checkbox("Bonus: use two reviewers for escalation")
        cols = st.columns(2)
        reviewer1 = cols[0].text_input("Reviewer 1", value="reviewer-1")
        reviewer2 = cols[1].text_input("Reviewer 2", value="reviewer-2")
        submitted = st.form_submit_button("Run review")
    if submitted:
        try:
            parse_pr_url(pr_url.strip())
            reviewers = [reviewer1.strip(), reviewer2.strip()] if multi else []
            if multi and (not all(reviewers) or len(set(reviewers)) != 2):
                raise ValueError("Enter two different reviewer names")
            new_id = str(uuid.uuid4())
            with st.spinner("Fetching PR and analyzing changes..."):
                result = asyncio.run(run_graph(pr_url.strip(), new_id,
                                                reviewers=reviewers, dry_run=dry_run))
            st.session_state.thread_id = new_id
            st.query_params["thread"] = new_id
            accept_result(result)
            st.rerun()
        except Exception as exc:
            st.error(f"Review failed: {exc}")

    if st.session_state.current.get("dry_run"):
        st.info("Dry run: the graph will produce a review without posting it.")
    pending = st.session_state.pending
    for item in pending:
        payload = item.value
        key = st.session_state.thread_id + "_" + item.id
        answer = (render_approval_card(payload, key) if payload["kind"] == "approval_request"
                  else render_escalation_card(payload, key))
        if answer is not None:
            try:
                with st.spinner("Resuming review..."):
                    result = asyncio.run(run_graph(st.session_state.pr_url,
                        st.session_state.thread_id, {item.id: answer}))
                accept_result(result)
                st.rerun()
            except Exception as exc:
                st.error(f"Resume failed: {exc}")

    final = st.session_state.final
    if final:
        action = final.get("final_action", "")
        if action == "commit_failed":
            st.error("Comment was not posted: " + final.get("error", "unknown error"))
        elif action == "rejected":
            st.warning("Rejected — no comment posted")
        elif action.startswith("dry_run_"):
            st.success("Dry run completed — no comment posted")
        elif action.startswith(("auto_", "committed")):
            st.success(f"Review posted — {action}")
            st.link_button("View review on GitHub", final.get("comment_url") or final["pr_url"])
        else:
            st.info(f"Session state: {action or 'not completed'}")
        if final.get("analysis"):
            st.metric("Final confidence", f"{final['analysis'].confidence:.0%}")
            st.markdown(final["analysis"].summary)
        if final.get("posted_comment_body"):
            with st.expander("Full review comment"):
                st.markdown(final["posted_comment_body"])

    thread = st.session_state.thread_id
    if thread:
        st.caption(f"thread_id = {thread}")
        st.code(f"uv run python -m audit.replay --thread {thread}", language="shell")
        with st.expander("Audit timeline"):
            st.dataframe(asyncio.run(replay_events(thread)), hide_index=True)
        with st.expander("Bonus: time-travel and compare"):
            current = st.session_state.current
            if current.get('source_thread'):
                source_history = asyncio.run(history_for(current['source_thread']))
                source = next((s for s in source_history if s.config['configurable']['checkpoint_id'] == current['source_checkpoint']), None)
                if source is not None:
                    st.caption('Source session: ' + current['source_thread'])
                    st.dataframe([
                        dict(version='Selected source checkpoint', confidence=source.values['analysis'].confidence,
                             summary=source.values['analysis'].summary, action=source.values.get('final_action')),
                        dict(version='Current fork', confidence=current['analysis'].confidence,
                             summary=current['analysis'].summary, action=current.get('final_action')),
                    ], hide_index=True)
            history = asyncio.run(history_for(thread))
            eligible = [s for s in history if s.values.get("analysis")]
            if eligible:
                chosen = st.selectbox("Historical checkpoint", range(len(eligible)),
                    format_func=lambda i: f"{eligible[i].created_at} | next={eligible[i].next}",
                    key="checkpoint_" + thread)
                snapshot = eligible[chosen]
                st.json({"checkpoint": snapshot.config["configurable"]["checkpoint_id"],
                         "analysis": snapshot.values["analysis"].model_dump(),
                         "action": snapshot.values.get("final_action")})
                st.caption("Fork the saved analysis into a new dry-run session; choose a different answer and compare. The source session stays available in Recent sessions.")
                if st.button("Fork checkpoint (dry run)", key="fork_" + thread):
                    try:
                        new_id, result = asyncio.run(fork_for(thread, snapshot.config["configurable"]["checkpoint_id"]))
                        st.session_state.thread_id = new_id
                        st.query_params["thread"] = new_id
                        accept_result(result)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Fork failed: {exc}")
                comparisons = [{"checkpoint": s.config["configurable"]["checkpoint_id"],
                                "confidence": s.values["analysis"].confidence,
                                "summary": s.values["analysis"].summary,
                                "action": s.values.get("final_action")} for s in eligible]
                st.dataframe(comparisons, hide_index=True)

    with st.expander("Bonus: confidence calibration"):
        report = asyncio.run(calibration())
        st.write({k: v for k, v in report.items() if k != "bins"})
        if report["bins"]:
            st.dataframe(report["bins"], hide_index=True)
        st.caption("One initial confidence and latest human approval decision per session. Positive gap means confidence exceeds approval frequency. This is a descriptive proxy, not a calibrated probability or proof of review quality; routed sessions are a biased sample.")


if __name__ == "__main__":
    main()

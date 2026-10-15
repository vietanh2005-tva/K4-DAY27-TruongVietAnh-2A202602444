"""Durable HITL review, structured audit, auto-edit and reviewer fan-out."""
from __future__ import annotations

import argparse
import asyncio
import json
import operator
import os
import time
import uuid
from typing import Annotated

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (AUTO_APPROVE_THRESHOLD, ESCALATE_THRESHOLD,
                            AuditEntry, PRAnalysis, ReviewState, risk_level_for)
from exercises.review_helpers import (approval_payload, review_messages,
                                      validate_answers, validate_choice)

console = Console()
AGENT_ID = 'pr-review-agent@v0.1'


class LabState(ReviewState, total=False):
    reviewers: list[str]
    reviewer_results: Annotated[list[dict], operator.add]
    dry_run: bool
    comment_url: str | None
    error: str | None
    source_thread: str | None
    source_checkpoint: str | None


async def audit(state, entry: AuditEntry):
    await write_audit_event(thread_id=state['thread_id'], pr_url=state['pr_url'], entry=entry)


async def log(state, action, started, decision='pending', reason=None,
              reviewer=None, analysis=None):
    a = analysis or state.get('analysis')
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action=action,
        confidence=a.confidence if a else 0.0,
        risk_level=risk_level_for(a.confidence) if a else 'med',
        reviewer_id=reviewer, decision=decision, reason=reason,
        execution_time_ms=int((time.monotonic() - started) * 1000)))


async def node_fetch_pr(state):
    t0 = time.monotonic()
    pr = await asyncio.to_thread(fetch_pr, state['pr_url'])
    await log(state, 'fetch_pr', t0, reason=f'Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}')
    return dict(pr_title=pr.title, pr_author=pr.author, pr_diff=pr.diff,
                pr_files=pr.files_changed, pr_head_sha=pr.head_sha)


async def node_analyze(state):
    t0 = time.monotonic()
    a = await get_llm().with_structured_output(PRAnalysis).ainvoke(review_messages(state))
    await log(state, 'analyze', t0, reason=a.confidence_reasoning, analysis=a)
    return {'analysis': a}


async def node_route(state):
    t0 = time.monotonic()
    c = state['analysis'].confidence
    decision = ('auto_approve' if c >= AUTO_APPROVE_THRESHOLD else
                'escalate' if c < ESCALATE_THRESHOLD else 'human_approval')
    await log(state, 'route', t0,
              {'auto_approve': 'auto', 'human_approval': 'pending', 'escalate': 'escalate'}[decision],
              reason=decision)
    return {'decision': decision}


async def node_human_approval(state):
    t0 = time.monotonic()
    await log(state, 'human_approval', t0, reason='Waiting for reviewer')
    response = validate_choice(interrupt(approval_payload(state)))
    await log(state, 'human_approval', t0, response['human_choice'],
              response['human_feedback'], os.environ.get('GITHUB_USER'))
    return response


def escalation_payload(state, reviewer=None):
    a = state['analysis']
    questions = a.escalation_questions or [
        'What production assumptions are missing from this diff?',
        'What security or migration tests cover these changes?']
    return dict(kind='escalation', pr_url=state['pr_url'], confidence=a.confidence,
                confidence_reasoning=a.confidence_reasoning, summary=a.summary,
                risk_factors=a.risk_factors, questions=questions, reviewer=reviewer)


async def node_escalate(state):
    t0 = time.monotonic()
    payload = escalation_payload(state)
    await log(state, 'escalate', t0, 'escalate', 'Waiting for answers')
    answers = validate_answers(interrupt(payload), payload['questions'])
    await log(state, 'escalate', t0, 'escalate', json.dumps(answers, ensure_ascii=False),
              os.environ.get('GITHUB_USER'))
    return {'escalation_answers': answers}


async def node_synthesize(state):
    t0 = time.monotonic()
    refined = await get_llm().with_structured_output(PRAnalysis).ainvoke(
        review_messages(state, state.get('escalation_answers') or {}))
    await log(state, 'synthesize', t0, 'escalate', refined.confidence_reasoning, analysis=refined)
    return {'analysis': refined}


async def node_auto_edit(state):
    t0 = time.monotonic()
    refined = await get_llm().with_structured_output(PRAnalysis).ainvoke(
        review_messages(state, {'requested_edit': state['human_feedback']}))
    await log(state, 'auto_edit', t0, 'edit', state['human_feedback'],
              os.environ.get('GITHUB_USER'), refined)
    return {'analysis': refined}


def _render_comment_body(state):
    a = state['analysis']
    lines = [f'### Automated review (confidence {a.confidence:.0%})', '', a.summary, '']
    for c in a.comments:
        lines.append(f'- **[{c.severity}]** `{c.file}:{c.line or "?"}` — {c.body}')
    if a.risk_factors:
        lines.extend(['', '**Risk factors:**'] + [f'- {risk}' for risk in a.risk_factors])
    if state.get('human_feedback'):
        lines.append(f'\nReviewer note: {state["human_feedback"]}')
    if state.get('escalation_answers'):
        lines.append('\n**Reviewer answered escalation questions:**')
        for q, answer in state['escalation_answers'].items():
            lines.append(f'> **{q}** {answer}')
    return '\n'.join(lines)


async def _post(state, label):
    body = _render_comment_body(state)
    if state.get('dry_run'):
        return dict(final_action='dry_run_' + label, posted_comment_body=body)
    try:
        comment_url = await asyncio.to_thread(post_review_comment, state['pr_url'], body)
    except Exception as exc:
        return dict(final_action='commit_failed', error=str(exc), posted_comment_body=body)
    return dict(final_action=label, posted_comment_body=body,
                comment_url=comment_url if isinstance(comment_url, str) else state['pr_url'])


async def node_commit(state):
    t0 = time.monotonic()
    choice = state.get('human_choice')
    if choice == 'reject':
        result = {'final_action': 'rejected'}
    elif choice in {'approve', 'edit'} or state.get('escalation_answers'):
        label = ('committed_after_edit' if choice == 'edit' else
                 'committed_after_escalation' if state.get('escalation_answers') else 'committed')
        result = await _post(state, label)
    else:
        result = {'final_action': 'rejected'}
    decision = choice or ('escalate' if state.get('escalation_answers') else 'reject')
    await log(state, 'commit', t0, decision, result.get('error') or result['final_action'],
              os.environ.get('GITHUB_USER') if choice or state.get('escalation_answers') else None)
    return result


async def node_auto_approve(state):
    t0 = time.monotonic()
    result = await _post(state, 'auto_committed')
    await log(state, 'auto_approve', t0, 'auto', result.get('error') or result['final_action'])
    return result


def build_graph(checkpointer):
    # The parent Send workers each drive a distinct durable reviewer thread.
    child = StateGraph(dict)

    def ask_reviewer(state):
        return {'payload': state['payload'],
                'answers': validate_answers(interrupt(state['payload']), state['payload']['questions'])}

    child.add_node('ask', ask_reviewer)
    child.add_edge(START, 'ask')
    child.add_edge('ask', END)
    reviewer_app = child.compile(checkpointer=checkpointer)

    async def reviewer_worker(state):
        reviewer = state['reviewer']
        child_id = f"{state['thread_id']}:reviewer:{reviewer}"
        cfg = {'configurable': {'thread_id': child_id, 'checkpoint_ns': ''}}
        snapshot = await reviewer_app.aget_state(cfg)
        if snapshot.values:
            result = snapshot.values
            pending = snapshot.next
        else:
            result = await reviewer_app.ainvoke({'payload': escalation_payload(state, reviewer)}, cfg)
            pending = bool(result.get('__interrupt__'))
        if pending:
            t0 = time.monotonic()
            await log(state, 'escalate', t0, 'escalate', f'Waiting for {reviewer}; thread={child_id}')
            payload = escalation_payload(state, reviewer)
            payload['reviewer_thread_id'] = child_id
            answer = validate_answers(interrupt(payload), payload['questions'])
            result = await reviewer_app.ainvoke(Command(resume=answer), cfg)
            await log(state, 'escalate', t0, 'escalate', json.dumps(answer), reviewer)
        return {'reviewer_results': [{'reviewer': reviewer, 'thread_id': child_id, 'answers': result['answers']}]}

    def dispatch(state):
        reviewers = state.get('reviewers') or []
        if len(reviewers) != 2 or len(set(reviewers)) != 2:
            raise ValueError('Fan-out requires exactly two different reviewer names')
        return [Send('reviewer', {**state, 'reviewer': reviewer}) for reviewer in reviewers]

    def collect(state):
        latest = {item['reviewer']: item for item in state.get('reviewer_results', [])}
        if set(latest) != set(state['reviewers']):
            raise ValueError('Both reviewer responses are required')
        answers = {f'[{reviewer}] {q}': a for reviewer, item in latest.items()
                   for q, a in item['answers'].items()}
        return {'escalation_answers': answers}

    g = StateGraph(LabState)
    nodes = dict(fetch_pr=node_fetch_pr, analyze=node_analyze, route=node_route,
                 auto_approve=node_auto_approve, human_approval=node_human_approval,
                 commit=node_commit, escalate=node_escalate, synthesize=node_synthesize,
                 auto_edit=node_auto_edit, fanout=lambda s: {}, reviewer=reviewer_worker,
                 collect=collect)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, 'fetch_pr')
    g.add_edge('fetch_pr', 'analyze')
    g.add_edge('analyze', 'route')

    def route(state):
        return 'fanout' if state['decision'] == 'escalate' and state.get('reviewers') else state['decision']

    g.add_conditional_edges('route', route,
                            {n: n for n in ('auto_approve', 'human_approval', 'escalate', 'fanout')})
    g.add_conditional_edges('human_approval', lambda s: 'auto_edit' if s.get('human_choice') == 'edit' else 'commit',
                            {'auto_edit': 'auto_edit', 'commit': 'commit'})
    g.add_edge('auto_edit', 'commit')
    g.add_edge('escalate', 'synthesize')
    g.add_conditional_edges('fanout', dispatch, ['reviewer'])
    g.add_edge('reviewer', 'collect')
    g.add_edge('collect', 'synthesize')
    g.add_edge('synthesize', 'commit')
    g.add_edge('auto_approve', END)
    g.add_edge('commit', END)
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload):
    console.print(Panel.fit(
        f"{payload['summary']}\n\n{payload['confidence_reasoning']}",
        title=f"{payload['kind']} conf={payload['confidence']:.0%}", border_style='yellow'))
    if payload['kind'] == 'approval_request':
        console.print(payload.get('comments', []))
        console.print(payload.get('diff_preview', ''))
        while True:
            choice = console.input('approve/reject/edit? ').strip().lower()
            if choice not in {'approve', 'reject', 'edit'}:
                continue
            feedback = console.input('Feedback: ').strip() if choice != 'approve' else ''
            if choice != 'edit' or feedback:
                return dict(choice=choice, feedback=feedback)
    console.print(payload.get('risk_factors', []))
    if payload.get('reviewer'):
        console.print(f"Reviewer: {payload['reviewer']}")
    answers = {}
    for question in payload['questions']:
        answer = ''
        while not answer:
            answer = console.input(f'Q: {question}\nA: ').strip()
        answers[question] = answer
    return answers


async def run(pr_url, thread_id=None, reviewers=None, dry_run=False):
    thread_id = thread_id or str(uuid.uuid4())
    console.print(f'thread_id = {thread_id}')
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {'configurable': {'thread_id': thread_id}}
        existing = await app.aget_state(cfg)
        if existing.values:
            if existing.values.get('pr_url') != pr_url:
                raise ValueError('This thread belongs to a different PR')
            result = dict(existing.values)
            pending = existing.interrupts
            if existing.next and not pending:
                result = await app.ainvoke(None, cfg)
                pending = result.get('__interrupt__', ())
        else:
            result = await app.ainvoke(dict(pr_url=pr_url, thread_id=thread_id,
                                           reviewers=reviewers or [], dry_run=dry_run), cfg)
            pending = result.get('__interrupt__', ())
        while pending:
            responses = {item.id: handle_interrupt(item.value) for item in pending}
            result = await app.ainvoke(Command(resume=responses), cfg)
            pending = result.get('__interrupt__', ())
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f'uv run python -m audit.replay --thread {thread_id}')
        return result


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument('--pr', required=True)
    p.add_argument('--thread')
    p.add_argument('--reviewers', nargs=2, metavar=('REVIEWER1', 'REVIEWER2'))
    p.add_argument('--dry-run', action='store_true', help='Build review without posting to GitHub')
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread, args.reviewers, args.dry_run))


if __name__ == '__main__':
    main()

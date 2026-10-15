"""Opt-in live dry run: real GitHub/LLM, explicitly simulated reviewer input.

Run: uv run python tests/live_verify.py --pr 1 --simulate-reviewer
No path in this script posts comments or represents simulated input as a person.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from audit.bonuses import calibration, checkpoint_history
from common.db import db_path, replay_events
from exercises.exercise_4_audit import build_graph


CONTEXT = (
    'VERIFICATION SIMULATION by coding assistant, not a human approval or a claim '
    'about author intent. Based only on the diff, deployment assumptions are unknown. '
    'Require a tested migration for existing databases, validated input and parameterized '
    'SQL; replace MD5 password hashing with a suitable password KDF, use server-issued '
    'random revocable tokens with protected storage, require HTTPS and authenticated '
    'user isolation, and add regression/security tests. These are recommendations, '
    'not assertions that changes or tests already exist. Keep unresolved findings.'
)


async def verify(number, thread=None):
    os.environ['GITHUB_USER'] = 'assistant-verification-simulation'
    thread = thread or 'live-dryrun-' + str(uuid4())
    print('thread_id:', thread, flush=True)
    url = f'https://github.com/VinUni-AI20k/PR-Demo/pull/{number}'
    cfg = {'configurable': {'thread_id': thread}}
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        snapshot = await graph.aget_state(cfg)
        if snapshot.values:
            if snapshot.values.get('pr_url') != url or not snapshot.values.get('dry_run'):
                raise ValueError('Only a matching dry-run session can be resumed')
            result = dict(snapshot.values)
            if snapshot.interrupts:
                result['__interrupt__'] = snapshot.interrupts
            elif snapshot.next:
                result = await graph.ainvoke(None, cfg)
        else:
            result = await graph.ainvoke(dict(pr_url=url, thread_id=thread, dry_run=True), cfg)
        initial = result['analysis'].model_dump()
        expected = 'human_approval' if number == 1 else 'escalate'
        if result.get('decision') != expected:
            raise AssertionError(f'Unexpected real routing: {result.get("decision")} ({initial["confidence"]})')
        interrupted = []
        while result.get('__interrupt__'):
            responses = {}
            for item in result['__interrupt__']:
                payload = item.value
                interrupted.append(payload)
                if payload['kind'] == 'approval_request':
                    responses[item.id] = {'choice': 'approve', 'feedback': 'VERIFICATION SIMULATION ONLY; not human approval.'}
                else:
                    responses[item.id] = {q: CONTEXT for q in payload['questions']}
            result = await graph.ainvoke(Command(resume=responses), cfg)
        assert result['final_action'].startswith('dry_run_'), result['final_action']
        history = await checkpoint_history(graph, thread)
    events = await replay_events(thread)
    report = dict(pr=number, pr_url=url, thread_id=thread, reviewer_input='simulated',
                  initial_analysis=initial, final_analysis=result['analysis'].model_dump(),
                  decision=result['decision'], final_action=result['final_action'],
                  interrupts=interrupted, checkpoints=len(history), events=events,
                  comment_body=result['posted_comment_body'], calibration=await calibration())
    folder = Path('scratch/live-check')
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'pr-{number}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (folder / f'pr-{number}-draft.md').write_text(result['posted_comment_body'], encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('pr', 'thread_id', 'decision', 'final_action', 'checkpoints')}, indent=2))
    print('confidence:', initial['confidence'], '->', result['analysis'].confidence)
    print('audit actions:', ', '.join(event['action'] for event in events))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pr', type=int, choices=(1, 2), required=True)
    parser.add_argument('--simulate-reviewer', action='store_true', required=True)
    parser.add_argument('--thread', help='Resume a previously interrupted/failed dry run')
    args = parser.parse_args()
    load_dotenv()
    asyncio.run(verify(args.pr, args.thread))

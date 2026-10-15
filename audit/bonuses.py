"""Read-only session reports and isolated time-travel experiments."""
from uuid import uuid4

from common.db import db_conn


async def recent_sessions():
    async with db_conn() as conn:
        async with conn.execute('''
            SELECT thread_id, pr_url, MIN(timestamp) AS started,
                   MAX(timestamp) AS last_event, COUNT(*) AS events,
                   CASE MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                        WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END AS worst_risk
            FROM audit_events GROUP BY thread_id, pr_url
            ORDER BY MAX(id) DESC LIMIT 25
        ''') as cursor:
            return [dict(row) for row in await cursor.fetchall()]


async def calibration():
    # One initial model score and the latest human decision per thread. Replayed
    # waiting events never count as responses; escalation is reported separately.
    async with db_conn() as conn:
        async with conn.execute('''
            WITH initial AS (
                SELECT thread_id, confidence FROM audit_events
                WHERE id IN (SELECT MIN(id) FROM audit_events WHERE action='analyze' GROUP BY thread_id)
            ), choices AS (
                SELECT thread_id, decision FROM audit_events
                WHERE id IN (SELECT MAX(id) FROM audit_events
                    WHERE action='human_approval' AND decision IN ('approve','reject','edit')
                      AND COALESCE(reviewer_id, '') <> 'assistant-verification-simulation'
                    GROUP BY thread_id)
            )
            SELECT initial.thread_id, confidence, choices.decision
            FROM initial LEFT JOIN choices USING(thread_id)
        ''') as cursor:
            rows = [dict(row) for row in await cursor.fetchall()]
    reviewed = [r for r in rows if r['decision'] is not None]
    approved = sum(r['decision'] == 'approve' for r in reviewed)
    rejected = sum(r['decision'] == 'reject' for r in reviewed)
    edited = sum(r['decision'] == 'edit' for r in reviewed)
    avg = sum(r['confidence'] for r in reviewed) / len(reviewed) if reviewed else None
    rate = approved / len(reviewed) if reviewed else None
    bins = []
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        group = [r for r in reviewed if lower <= r['confidence'] < lower + .2
                 or (lower == .8 and r['confidence'] == 1)]
        if group:
            bins.append(dict(bucket=f'{lower:.0%}–{lower + .2:.0%}', sessions=len(group),
                             mean_confidence=sum(r['confidence'] for r in group) / len(group),
                             approval_rate=sum(r['decision'] == 'approve' for r in group) / len(group)))
    return dict(total_sessions=len(rows), reviewed=len(reviewed), approved=approved,
                rejected=rejected, edited=edited, mean_confidence=avg, approval_rate=rate,
                gap=avg-rate if reviewed else None, bins=bins)


async def checkpoint_history(app, thread_id):
    cfg = {'configurable': {'thread_id': thread_id}}
    return [snapshot async for snapshot in app.aget_state_history(cfg)]


async def fork_checkpoint(app, source_thread, checkpoint_id):
    """Branch a prior review at routing, with external posting disabled.

    The historical checkpoint supplies the analysis/context. Starting after
    route makes both HITL interrupts fire afresh, with new reviewer child threads.
    """
    history = await checkpoint_history(app, source_thread)
    snapshot = next((s for s in history if s.config['configurable']['checkpoint_id'] == checkpoint_id), None)
    if snapshot is None or not snapshot.values.get('analysis'):
        raise ValueError('Select a checkpoint with a completed analysis')
    new_id = str(uuid4())
    values = dict(snapshot.values)
    for key in ('human_choice', 'human_feedback', 'escalation_answers',
                'posted_comment_body', 'final_action', 'comment_url', 'error'):
        values[key] = None
    values.update(thread_id=new_id, dry_run=True, reviewer_results=[],
                  source_thread=source_thread, source_checkpoint=checkpoint_id)
    cfg = {'configurable': {'thread_id': new_id}}
    # Resume from the selected historical analysis, re-route to allow a different answer.
    await app.aupdate_state(cfg, values, as_node='analyze')
    result = await app.ainvoke(None, cfg)
    return new_id, result

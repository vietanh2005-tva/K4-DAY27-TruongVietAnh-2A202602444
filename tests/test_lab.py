"""Offline integration tests using real LangGraph and SQLite."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from audit.bonuses import calibration, checkpoint_history, fork_checkpoint, recent_sessions
from common.db import replay_events
from common.schemas import PRAnalysis
from exercises import exercise_1_confidence as e1, exercise_2_hitl as e2
from exercises import exercise_3_escalation as e3, exercise_4_audit as e4

PR = 'https://github.com/test/demo/pull/1'
FETCHED = SimpleNamespace(title='Feature', author='author', diff='+ change', files_changed=['app.py'], head_sha='1234567')


def analysis(score=.65, summary='Initial'):
    return PRAnalysis(summary=summary, confidence=score, confidence_reasoning='Missing context',
                      risk_factors=['Migration'], escalation_questions=['How is app.py migrated?', 'What tests cover app.py?'])


def llm(*outputs):
    model = MagicMock()
    model.with_structured_output.return_value = model
    model.invoke.side_effect = list(outputs)
    model.ainvoke = AsyncMock(side_effect=list(outputs))
    return model


class BasicExercises(unittest.TestCase):
    def test_routing_boundaries(self):
        for score, branch in [(.579, 'escalate'), (.58, 'human_approval'), (.729, 'human_approval'), (.73, 'auto_approve')]:
            with self.subTest(score=score), patch.object(e1, 'fetch_pr', return_value=FETCHED), patch.object(e1, 'get_llm', return_value=llm(analysis(score))):
                result = e1.build_graph().invoke({'pr_url': PR})
                self.assertEqual(result['decision'], branch)

    def test_approval_reject_edit(self):
        for choice in ('approve', 'reject', 'edit'):
            with self.subTest(choice=choice), patch.object(e2, 'fetch_pr', return_value=FETCHED), patch.object(e2, 'get_llm', return_value=llm(analysis(), analysis(.8, 'Edited'))), patch.object(e2, 'post_review_comment') as post:
                app = e2.build_graph()
                cfg = {'configurable': {'thread_id': choice}}
                result = app.invoke({'pr_url': PR}, cfg)
                self.assertIn('__interrupt__', result)
                self.assertFalse(post.called)
                result = app.invoke(Command(resume={'choice': choice, 'feedback': 'Add tests'}), cfg)
                self.assertEqual(post.call_count, 0 if choice == 'reject' else 1)
                self.assertNotIn('__interrupt__', result)
                if choice == 'edit':
                    self.assertIn('Edited', post.call_args.args[1])

    def test_escalation(self):
        with patch.object(e3, 'fetch_pr', return_value=FETCHED), patch.object(e3, 'get_llm', return_value=llm(analysis(.4), analysis(.8, 'Refined'))), patch.object(e3, 'post_review_comment') as post:
            app = e3.build_graph()
            cfg = {'configurable': {'thread_id': 'escalation'}}
            result = app.invoke({'pr_url': PR}, cfg)
            questions = result['__interrupt__'][0].value['questions']
            result = app.invoke(Command(resume={q: 'Tested' for q in questions}), cfg)
            self.assertEqual(result['final_action'], 'committed_after_escalation')
            self.assertIn('Refined', post.call_args.args[1])


class DurableExercises(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / 'test.db')
        self.env = patch.dict(os.environ, HITL_DB_PATH=self.path, GITHUB_USER='tester')
        self.env.start()
        self.fetch = patch.object(e4, 'fetch_pr', return_value=FETCHED)
        self.fetch.start()
        self.post = patch.object(e4, 'post_review_comment')
        self.mock_post = self.post.start()

    async def asyncTearDown(self):
        self.post.stop()
        self.fetch.stop()
        self.env.stop()
        self.temp.cleanup()

    async def invoke(self, thread, value=None, *, resume=False, **extra):
        async with AsyncSqliteSaver.from_conn_string(self.path) as cp:
            await cp.setup()
            return await e4.build_graph(cp).ainvoke(Command(resume=value) if resume else dict(pr_url=PR, thread_id=thread, **extra), {'configurable': {'thread_id': thread}})

    async def test_durable_approval_edit_audit_calibration(self):
        for choice in ('approve', 'reject', 'edit'):
            with patch.object(e4, 'get_llm', return_value=llm(analysis(), analysis(.8, 'Edited'))):
                result = await self.invoke(choice)
                self.assertIn('__interrupt__', result)
                result = await self.invoke(choice, {'choice': choice, 'feedback': 'Improve tests'}, resume=True)
                self.assertEqual(result['final_action'], {'approve': 'committed', 'reject': 'rejected', 'edit': 'committed_after_edit'}[choice])
                events = await replay_events(choice)
                self.assertTrue({'fetch_pr', 'analyze', 'route', 'human_approval', 'commit'} <= {e['action'] for e in events})
                response = [e for e in events if e['action'] == 'human_approval' and e['decision'] == choice]
                self.assertEqual(len(response), 1)
                self.assertEqual(response[0]['reviewer_id'], 'tester')
        self.assertEqual(self.mock_post.call_count, 2)
        report = await calibration()
        self.assertEqual((report['reviewed'], report['approved'], report['edited'], report['rejected']), (3, 1, 1, 1))

    async def test_auto_dry_run_failure(self):
        for name, dry in [('auto', False), ('preview', True), ('failure', False)]:
            with patch.object(e4, 'get_llm', return_value=llm(analysis(.9))):
                self.mock_post.side_effect = RuntimeError('HTTP 403') if name == 'failure' else None
                result = await self.invoke(name, dry_run=dry)
                self.assertEqual(result['final_action'], {'auto': 'auto_committed', 'preview': 'dry_run_auto_committed', 'failure': 'commit_failed'}[name])
        self.assertEqual(self.mock_post.call_count, 2)
        self.assertFalse(any(e['decision'] == 'auto' and e['risk_level'] == 'high' for e in await replay_events('auto')))

    async def test_single_escalation(self):
        with patch.object(e4, 'get_llm', return_value=llm(analysis(.4), analysis(.8, 'Refined'))):
            result = await self.invoke('single')
            payload = result['__interrupt__'][0].value
            result = await self.invoke('single', {q: 'Tested' for q in payload['questions']}, resume=True)
            self.assertEqual(result['final_action'], 'committed_after_escalation')
            self.assertEqual((await recent_sessions())[0]['worst_risk'], 'high')

    async def test_calibration_excludes_simulated_reviewer(self):
        with patch.dict(os.environ, GITHUB_USER='assistant-verification-simulation'), patch.object(e4, 'get_llm', return_value=llm(analysis())):
            await self.invoke('simulated', dry_run=True)
            await self.invoke('simulated', {'choice': 'approve'}, resume=True)
            report = await calibration()
            self.assertEqual(report['total_sessions'], 1)
            self.assertEqual(report['reviewed'], 0)
            self.assertIsNone(report['approval_rate'])

    async def test_two_reviewers_wait_for_both(self):
        with patch.object(e4, 'get_llm', return_value=llm(analysis(.4), analysis(.8, 'Both answered'))):
            result = await self.invoke('multi', reviewers=['alice', 'bob'])
            self.assertEqual(len(result['__interrupt__']), 2)
            first, second = result['__interrupt__']
            result = await self.invoke('multi', {first.id: {q: 'Alice' for q in first.value['questions']}}, resume=True)
            self.assertTrue(result['__interrupt__'])
            self.assertEqual(self.mock_post.call_count, 0)
            result = await self.invoke('multi', {second.id: {q: 'Bob' for q in second.value['questions']}}, resume=True)
            self.assertEqual(result['final_action'], 'committed_after_escalation')
            self.assertEqual(len(result['reviewer_results']), 2)
            self.assertEqual(self.mock_post.call_count, 1)
            async with AsyncSqliteSaver.from_conn_string(self.path) as cp:
                await cp.setup()
                for reviewer in ('alice', 'bob'):
                    child = await cp.aget_tuple({'configurable': {'thread_id': f'multi:reviewer:{reviewer}'}})
                    self.assertIsNotNone(child)

    async def test_time_travel_isolated_no_post(self):
        with patch.object(e4, 'get_llm', return_value=llm(analysis())):
            await self.invoke('source')
            async with AsyncSqliteSaver.from_conn_string(self.path) as cp:
                await cp.setup()
                app = e4.build_graph(cp)
                history = await checkpoint_history(app, 'source')
                checkpoint = next(s for s in history if s.values.get('analysis'))
                fork_id, result = await fork_checkpoint(app, 'source', checkpoint.config['configurable']['checkpoint_id'])
                self.assertNotEqual(fork_id, 'source')
                self.assertIn('__interrupt__', result)
                result = await app.ainvoke(Command(resume={'choice': 'approve'}), {'configurable': {'thread_id': fork_id}})
                self.assertEqual(result['final_action'], 'dry_run_committed')
                self.assertEqual(self.mock_post.call_count, 0)
                self.assertTrue((await app.aget_state({'configurable': {'thread_id': 'source'}})).interrupts)


if __name__ == '__main__':
    unittest.main()

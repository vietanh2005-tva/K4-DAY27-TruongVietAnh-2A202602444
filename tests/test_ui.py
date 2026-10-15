"""Streamlit's real test runner with external services mocked."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from streamlit.testing.v1 import AppTest
from common.schemas import PRAnalysis
from exercises import exercise_4_audit as e4


class UITest(unittest.TestCase):
    def test_escalation_and_restore_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            model = MagicMock()
            model.with_structured_output.return_value = model
            model.ainvoke = AsyncMock(side_effect=[
                PRAnalysis(summary='Security changes', confidence=.4, confidence_reasoning='Need context', risk_factors=['MD5'], escalation_questions=['Why MD5?', 'Which tests?']),
                PRAnalysis(summary='Refined review', confidence=.7, confidence_reasoning='Context provided')])
            pr = SimpleNamespace(title='Auth', author='author', diff='+ md5(password)', files_changed=['auth.py'], head_sha='abc1234')
            with patch.dict(os.environ, HITL_DB_PATH=str(Path(directory) / 'ui.db')), patch.object(e4, 'fetch_pr', return_value=pr), patch.object(e4, 'get_llm', return_value=model), patch.object(e4, 'post_review_comment') as post:
                path = str(Path(__file__).resolve().parents[1] / 'app.py')
                app = AppTest.from_file(path, default_timeout=30).run()
                app.text_input[0].set_value('https://github.com/test/demo/pull/2')
                next(b for b in app.button if b.label == 'Run review').click().run()
                self.assertEqual(len(app.exception), 0)
                thread = app.session_state['thread_id']
                restored = AppTest.from_file(path, default_timeout=30)
                restored.query_params['thread'] = thread
                restored.run()
                self.assertEqual(len(restored.exception), 0)
                self.assertTrue(any('Strong escalation' in h.value for h in restored.subheader))
                for field in restored.text_input:
                    if field.label in {'Why MD5?', 'Which tests?'}:
                        field.set_value('Use Argon2; add regression tests')
                next(b for b in restored.button if b.label == 'Submit answers').click().run()
                self.assertEqual(len(restored.exception), 0)
                self.assertTrue(any('Dry run completed' in h.value for h in restored.success))
                self.assertEqual(model.ainvoke.await_count, 2)
                post.assert_not_called()

    def test_start_approve_and_no_duplicate_on_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            model = MagicMock()
            model.with_structured_output.return_value = model
            model.ainvoke = AsyncMock(return_value=PRAnalysis(summary='Mock review', confidence=.65, confidence_reasoning='Need a reviewer'))
            pr = SimpleNamespace(title='Mock', author='author', diff='+ change', files_changed=['a.py'], head_sha='abc1234')
            with patch.dict(os.environ, HITL_DB_PATH=str(Path(directory) / 'ui.db')), patch.object(e4, 'fetch_pr', return_value=pr), patch.object(e4, 'get_llm', return_value=model), patch.object(e4, 'post_review_comment') as post:
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
                self.assertEqual(len(app.exception), 0)
                app.text_input[0].set_value('https://github.com/test/demo/pull/1')
                next(b for b in app.button if b.label == 'Run review').click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any('Approval requested' in s.value for s in app.subheader))
                next(b for b in app.button if b.label == 'Approve').click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any('Dry run completed' in s.value for s in app.success))
                app.run()
                self.assertEqual(len(app.exception), 0)
                source = app.session_state['thread_id']
                next(b for b in app.button if b.label == 'Fork checkpoint (dry run)').click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertNotEqual(app.session_state['thread_id'], source)
                self.assertEqual(app.session_state['current']['source_thread'], source)
                next(b for b in app.button if b.label == 'Reject').click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.session_state['final']['final_action'], 'rejected')
                post.assert_not_called()

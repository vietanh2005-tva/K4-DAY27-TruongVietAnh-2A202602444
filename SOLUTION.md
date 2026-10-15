# Completed Day27 lab

The four exercises and Streamlit app are implemented. Thresholds are imported
from `common/schemas.py`: auto >= 0.73, escalation < 0.58, approval otherwise.
`PRAnalysis` and the supplied audit schema remain unchanged.

## Setup and run (PowerShell)

```powershell
uv sync
# If .env does not exist yet:
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY`, `AI_PROVIDER=openai`, `MODEL_NAME=gpt-6-luna`, and
`OPENAI_BASE_URL=https://modelapi.vn/v1`. Also set `GITHUB_TOKEN` (classic
`repo` scope for the private demo repository) and `GITHUB_USER` (your GitHub
username). Do not commit `.env` or local database files.

```powershell
uv run python exercises/exercise_1_confidence.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
uv run python exercises/exercise_1_confidence.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
uv run python exercises/exercise_2_hitl.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
uv run python exercises/exercise_3_escalation.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
uv run python exercises/exercise_4_audit.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1 --dry-run
uv run streamlit run app.py
```

Exercise 2/3 post actual GitHub discussion comments when approved, auto-approved,
edited, or synthesized after escalation. They do not submit a formal GitHub
Approve review. Exercise 4 adds `--dry-run`; Streamlit defaults to dry run.
Uncheck it for the lab's actual posting demonstration. Rejected reviews never post.
Failures display `commit_failed`, not a success message.

## Audit and durable sessions

```powershell
uv run python -m audit.replay --list
uv run python -m audit.replay --thread THREAD_ID
uv run python exercises/exercise_4_audit.py --pr PR_URL --thread THREAD_ID
```

`--thread` restores an existing checkpoint rather than fetching and analyzing
again. In Streamlit, Recent sessions restores the full saved state and pending
interrupts. The URL's `?thread=...` parameter also restores after browser refresh.
Both SQLite checkpoints and audit events use `HITL_DB_PATH` (default
`hitl_audit.db`). The waiting audit entry repeats when a node restarts on resume,
as described in the lab; human response entries are recorded after the interrupt.
Risk ordering uses explicit high/medium/low ranks, not string ordering.

## All four bonuses

1. **Time-travel:** expand the app's time-travel panel, inspect the history from
   `aget_state_history`, select an analyzed checkpoint, and fork it. The historical
   analysis/context is copied into a new durable session and routing runs again,
   so you can supply a different reviewer answer. Forks always use dry run and
   fresh reviewer child IDs. The source session is preserved; compare confidence,
   summary, action, and audit timeline across the source/fork sessions.
2. **Confidence calibration:** expand the calibration panel. It joins one initial
   analysis and the latest actual human choice per thread, then reports average
   confidence, approve/reject/edit counts, approval rate, gap, and confidence bins.
   Pending/replayed waiting events are excluded. This is a descriptive proxy:
   review completeness confidence and human approval measure different things;
   only routed, human-reviewed sessions enter the approval comparison.
3. **Multi-reviewer fan-out:** check the two-reviewer option in Streamlit and enter
   distinct names, or run:

   ```powershell
   uv run python exercises/exercise_4_audit.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2 --reviewers alice bob --dry-run
   ```

   `Send` dispatches parallel reviewer workers. Each has a separate durable child
   thread `<parent>:reviewer:<name>`. Each form resumes its own interrupt ID; the
   graph waits for both, aggregates attributed answers, then synthesizes once.
   This lab UI lets an operator enter both reviewers' responses; it is not an
   authentication or permission system for two remote users.
4. **Auto-edit:** select Edit and provide feedback. A separate LLM node rewrites
   the review using the original diff, original analysis and feedback, then the
   downstream commit node posts it (or previews it in dry run). Audit records
   the edit and new confidence. Exercise 2 and 3 also support this behavior.

## Verification

```powershell
uv run python -B -m unittest discover -s tests -v
```

Tests use real LangGraph, SQLite checkpoints and Streamlit AppTest, mocking only
GitHub/LLM services. They cover thresholds, approve/reject/edit, escalation,
restart/resume, structured audit, calibration deduplication, risk ordering,
posting failures, dry runs, isolated checkpoint forks, waiting for both reviewers,
UI decisions and restoring a session after reload. They never post to GitHub.

The configured modelapi.vn endpoint was also checked with a synthetic README
diff and returned a valid structured `PRAnalysis`. Actual demo PR verification
still requires a GitHub token and repository access; it must not be inferred
from offline test results.

For an opt-in live dry run against the public demo PRs, use:

```powershell
uv run python tests/live_verify.py --pr 1 --simulate-reviewer
uv run python tests/live_verify.py --pr 2 --simulate-reviewer
```

These commands call the actual GitHub/LLM services, persist checkpoints and audit
events locally, and save reports/drafts under git-ignored `scratch/live-check/`.
Reviewer inputs are explicitly labeled simulations and attributed to
`assistant-verification-simulation`, not the human username. Calibration excludes
those simulated approval decisions. No GitHub comment is posted by these checks.

The shared prompt distinguishes confidence in finding a defect from confidence
that a review can proceed without human context. Bounded additive schema changes
with migration questions use the middle band; unresolved authentication, identity,
secret-handling or injection risks use escalation. The routing thresholds remain
0.73 and 0.58; no PR number or URL is used to force the model's score.

For submission, keep the repository Public with the required personal name,
verify real demo PR comments appear under your username, run from a clean clone,
then submit your own repository URL on VLearn.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module architecture

The project is built module-by-module (see README for the list). Each module lives in `src/<module>/` and follows the same shape, established by `src/nlp/` and `src/alerts/`:

- A `*Service` class owning its DB access directly (`AsyncSessionLocal()`), with `run()`/`stop()` for the continuous poll loop and a one-shot method (e.g. `process_pending()`, `score_pending()`) callable from a CLI flag.
- Pure computation split into separate files with no DB/IO (e.g. `analyzer.py`, `features.py`, `scorer.py`) — never mix DB access into these.
- A companion class that owns its own DB session but has no loop of its own, invoked from another service's tick (`SnapshotScheduler` invoked by NLP, `OutcomeLabeler` invoked by ML) — use this shape instead of adding a new top-level `asyncio.gather()` task when the work is a lightweight companion step to an existing service.
- `src/<module>/__init__.py` eager-imports pure/lightweight pieces and lazy-imports (`__getattr__`) anything depending on heavy libs (sqlalchemy, loguru, sklearn/xgboost), so unit tests that only need the pure logic don't pull them in.
- Wired into `main.py`: instantiate in `run_all()`, add to the `asyncio.gather(...)` and the `finally` stop block, plus a `--<module>-once` argparse flag with a matching one-shot coroutine (see `nlp_once`/`ml_once` for the pattern).

**When adding any new service module that imports `AsyncSessionLocal` directly** (either the service itself or a companion class like `OutcomeLabeler`), you must add a `monkeypatch.setattr(<module>, "AsyncSessionLocal", TestSession)` line for it in `tests/conftest.py`'s `test_db` fixture — one line per module that has its own `from ..core.database import AsyncSessionLocal` import. Easy to miss; tests will silently hit the real DB path (or fail confusingly) if skipped.

`requirements.txt` stages a module's heavier deps as commented-out lines under a `# ── <Module Name> ──` header before the module is built (e.g. scikit-learn/xgboost were commented until Module 4). When building that module, uncomment them rather than adding fresh entries.

## Known repo state

- **`.claude/worktrees/nlp-module` (branch `worktree-nlp-module`) is stray, unmerged exploration** — an earlier messy attempt at ML scoring (duplicated v1/v2 files) that drifted into a partial Module 6 dashboard. Don't treat it as a reference or a starting point without checking with the user first.
- No `.gitignore` exists yet — `__pycache__/*.pyc` files are currently tracked in git. Don't `git add -A` blindly; stage files explicitly.
- No CI, no linter/formatter config (no ruff/black/eslint). Don't assume a `lint` or `format` command exists.
- Known pre-existing bug (unrelated to any specific module): `SnapshotScheduler`/NLP processing can throw `can't compare offset-naive and offset-aware datetimes` because SQLite doesn't round-trip `tzinfo` on `Tweet.tweeted_at`. Not yet fixed.

## Testing

- Full suite: `pytest` (config in `pytest.ini` — coverage report + HTML output to `data/coverage` on every run).
- Tests are split `tests/unit/` (mocked/`SimpleNamespace`-based, no real DB for pure logic) and `tests/integration/test_pipeline.py` (real temp SQLite via the `test_db` fixture, chains multiple modules together).

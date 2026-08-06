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
- `.gitignore` exists (`__pycache__`, `.env`, `data/*.db`, `logs/*.log`, `models/*.joblib`, `CLAUDE.local.md`). `ruff.toml` configures lint (E/F/W/I) — a `PostToolUse`/`Write|Edit` hook in `.claude/settings.json` runs `ruff check --fix` automatically on every `.py` file edited.
- `DateTime(timezone=True)` columns don't round-trip `tzinfo` on SQLite — use `AwareDateTime` (`src/core/database.py`) for any new datetime column, never the raw SQLAlchemy type. `_make_engine()` also sets `PRAGMA journal_mode=WAL` + `busy_timeout=30000` (needed once more than one service writes concurrently — collector, scheduler, nlp, alerts, ml all do).

### There is a real production deployment, separate from this repo's history

A Proxmox VM on the user's home LAN (`ssh ubuntu@192.168.1.67`, hostname `VM101`) runs the **actual live bot** — real Telegram alerts, a Flask dashboard on port 5000, Postgres in Docker. Its `~/Morpheus` git checkout is on `main`, **28 commits ahead of `origin/main`, never pushed, and unrelated to this local repo's commit history** (this repo's Modules 1-5 were rebuilt from scratch against `MockTwitterClient` without knowing the VM existed). The VM's commits contain real production learnings: Twitter/Nitter died, replaced with a **StockTwits** collector (ported into this repo 2026-08-06 — `src/collector/stocktwits_client.py`, `config/symbols.yaml`), account reliability ranking, Telegram digest formatting, a short-seller report collector, plus a messy uncommitted "Phase 2 hybrid ML" experimental layer (not ported — this repo's `src/ml/` covers the same need more cleanly).

**Before building anything collector/alerts/ML-related, check whether the VM already solved it** — SSH in and read the code first rather than re-deriving from scratch. Don't touch the VM's running processes or deploy to it without the user explicitly asking for that as a separate step.

**Known issue (2026-08-06, unresolved):** StockTwits' public API returns 403 Forbidden on every symbol — confirmed both locally and from the VM itself, and the VM's live logs show it's been silently failing (no crash, just zero messages collected) since at least that morning. Likely new bot detection on StockTwits' side. Check current status before assuming the StockTwits collector works.

**Keep this file (and the VM's own `~/Morpheus/CLAUDE.md`) up to date as a standing habit** — when something notable happens (a bug found, a decision made, an architecture note worth knowing next session), write it here. Same for the VM's CLAUDE.md when the finding is specific to production/that deployment.

## Testing

- Full suite: `pytest` (config in `pytest.ini` — coverage report + HTML output to `data/coverage` on every run).
- Tests are split `tests/unit/` (mocked/`SimpleNamespace`-based, no real DB for pure logic) and `tests/integration/test_pipeline.py` (real temp SQLite via the `test_db` fixture, chains multiple modules together).

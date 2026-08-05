---
name: new-module
description: Scaffold a new financial-radar module (src/<name>/) following the established service pattern — use when starting a new numbered module (e.g. "Module 6 — Dashboard") from scratch.
disable-model-invocation: true
---

Scaffold a new module for financial-radar, given `$ARGUMENTS` (module name/number and one-line purpose, e.g. "Module 6 — Dashboard: FastAPI + React PWA").

Before writing any code, always: read `src/nlp/` and `src/ml/` in full as the reference implementations, check `git worktree list` and flag `.claude/worktrees/nlp-module` if it's relevant prior art for this module (confirm with the user whether to inspect/salvage it or build fresh — don't assume either), and propose a short plan (EnterPlanMode) before implementing since this always touches multiple files.

## Files to create

- `src/<module>/__init__.py` — eager-import pure/lightweight pieces; lazy-import (`__getattr__`) anything needing sqlalchemy/loguru/heavy deps, matching `src/ml/__init__.py`.
- Pure computation files (no DB/IO) for any non-trivial logic — mirrors `src/nlp/analyzer.py` / `src/ml/features.py`.
- `src/<module>/service.py` — a `*Service` class: `run()`/`stop()` loop, plus a one-shot method callable from a CLI flag. If the module needs a lightweight companion step (not its own full service), give it its own DB-owning class invoked from the parent service's tick instead of a new top-level `asyncio.gather()` task — mirrors `SnapshotScheduler` (invoked by NLP) and `OutcomeLabeler` (invoked by ML).

## Wiring checklist (easy to miss any of these)

1. `src/core/database.py` — add any new ORM columns/tables needed (ask first if it's a schema change to existing tables).
2. `src/core/settings.py` — new settings under a `# ── <Module> ──` block, matching the `ml_*`/`alert_*` style. Only promote to `Settings` what's actually meant to be tuned per-deployment; small internal constants stay as module-level `_CONSTANTS` like `_BATCH_SIZE`.
3. `.env.example` — matching block for any new settings.
4. `requirements.txt` — if the module needs deps not yet installed, uncomment its staged block (or add one) rather than leaving them commented once you actually build against them.
5. `main.py` — instantiate the service in `run_all()`, add to `asyncio.gather(...)` and the `finally` stop block; add a `--<module>-once` flag with a matching one-shot coroutine (see `ml_once`/`nlp_once`).
6. **`tests/conftest.py`** — for every module file that does `from ..core.database import AsyncSessionLocal` at module level (the service itself, and any companion class), add a `monkeypatch.setattr(<module>, "AsyncSessionLocal", TestSession)` line inside the `test_db` fixture. Miss this and DB-touching tests fail confusingly or hit the wrong session.

## Tests

- `tests/unit/test_<module>.py` — pure logic with `SimpleNamespace` fakes (no DB), DB-touching pieces via the `test_db` fixture. Follow the class-per-concern layout used in `tests/unit/test_ml.py` / `tests/unit/test_alerts.py`.
- A small addition (2-3 tests, not a re-test of unit cases) to `tests/integration/test_pipeline.py`: one end-to-end smoke test chaining this module with whatever upstream module feeds it, and — if this module is meant to stay decoupled from another one — a smoke test asserting it doesn't touch that other module's fields/behavior (see `TestMLModule::test_ml_module_does_not_touch_alerts` for the pattern).

## Verification

Run `pytest` (full suite must stay green) and the new `--<module>-once` CLI flag against a fresh `--init-db` before considering the module done — same sequence used for Module 4 (`--init-db && --poll-once && --nlp-once && --<module>-once`).

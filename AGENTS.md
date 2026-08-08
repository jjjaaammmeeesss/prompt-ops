# AGENTS

Instructions for AI agents working in this repository.

## Knowledge store

- **`docs/solutions/`** — Documented solutions to past problems, organized by category. Each file has YAML frontmatter with `problem_type`, `tags`, `applies_when`, and other searchable fields. Read relevant docs before solving a problem that may have been solved before.
- **`CONCEPTS.md`** — Project-specific vocabulary. Defines terms like MIPROv2, LLM-as-Judge Metric, and weight redistribution that have precise meanings in this codebase. Read when encountering an unfamiliar domain term.

## Project structure

- `src/prompt_ops/` — Core Python package (prompt optimization framework)
- `use-cases/` — Example use cases demonstrating prompt-ops on real tasks
- `notebook/` — Tutorial notebooks
- `frontend/` — React + TypeScript web frontend
- `frontend/backend/` — FastAPI Python backend (serves the frontend)

## Conventions

- Python >= 3.10
- TypeScript 5.5.3 for frontend
- Prompt files carry version numbers in filename and internal title that must match
- **Executors are version-linked to the prompt they run** (hard rule): the executor filename must carry the prompt's version it adapts (e.g. `system_prompt_v3.1.txt` → `popup_generator_v31.py`; a generic name like `popup_generator.py` is only allowed for one pinned prompt version). Each executor must have its own `__version__` (its own iteration counter, independent of the prompt version), plus a `PROMPT_VERSION` field that equals the adapted prompt version and stays two-way aligned with the prompt file's internal title. Changing the prompt version requires creating/renaming the matching executor or updating its `PROMPT_VERSION` in the same commit.
- `config.yaml` in each use-case wires system prompt, dataset, model, and metric

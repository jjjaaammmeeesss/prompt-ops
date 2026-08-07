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
- `config.yaml` in each use-case wires system prompt, dataset, model, and metric

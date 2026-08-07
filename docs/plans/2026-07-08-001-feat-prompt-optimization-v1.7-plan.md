---
title: "Prompt Optimization v1.7 with Extended Dataset - Plan"
type: feat
date: 2026-07-08
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
product_contract_source: ce-plan-bootstrap
---

## Goal Capsule

- **Objective**: Use prompt-ops MIPROv2 pipeline to optimize the v1.7 parent-child coach prompt, replacing the insufficient 14-example dataset with an extended dataset built from 82 new test case files (22 from first batch + 60 from second batch) plus existing examples.
- **Authority hierarchy**: The v1.7 system prompt (values, methodology, output schema) is authoritative for golden-answer generation; prompt-ops config.yaml is authoritative for optimization parameters.
- **Stop conditions**: Optimization completes with MIPROv2 returning a best program AND the optimized prompt is saved to `results/` AND evaluation scores show meaningful differentiation across trials (not all identical) AND the optimized prompt outperforms the v1.7 baseline on the test set (14 hand-labeled examples).
- **Execution profile**: Standard — data conversion pipeline, prompt-ops optimization run, result validation.
- **Tail ownership**: The optimized prompt artifact lives in `use-cases/parent-child-coach/results/`; the dataset is reusable for future optimization runs.

## Product Contract

### Summary

Extend the prompt-ops dataset from 14 to ~96 examples using the user's 82 new test case files (22 from the first batch at `C:\Users\h\Downloads\7.6 test_cases\test_cases\` + 60 from the second batch at `C:\Users\h\Downloads\第二批-7.6（60）\`), generate golden answers via a stronger oracle model, and re-run MIPROv2 optimization with sufficient data to produce meaningful score differentiation.

### Problem Frame

The previous optimization run (2026-07-08) completed technically but produced no meaningful improvement: all 11 MIPROv2 trials scored identically at 66.67% (2/3) because the 14-example dataset was split into only 3 train / 3 validation / 8 test examples. The validation set of 3 is too small for the metric to differentiate between instruction candidates. However, this root cause (small dataset) has **not been empirically validated** — it is possible the flat scores stem from a metric sensitivity issue or DSPy configuration problem rather than dataset size alone. The user has now provided 82 additional test case files spanning 5 difficulty levels and 22 scenario categories with 2-6 variants each — enough to build a dataset that lets MIPROv2 distinguish good configurations from bad ones, provided the root cause is indeed dataset size.

### Requirements

**Data conversion**
- R1. Convert 82 plain-text test case files into the prompt-ops dataset JSON format with `question` fields (answer generation is covered by R3).
- R2. Preserve the dialogue text verbatim as the `question` field; strip metadata headers (`# 难度: ...` lines).
- R3. Generate `answer` fields (golden JSON with `should_popup`, `tone`, `popup_insight`, `popup_suggestion`) for each test case using a stronger oracle model (see KTD1 for rationale).

**Dataset assembly**
- R4. Merge the 82 new examples with the existing 14 examples into a single dataset of ~96 examples.
- R5. Configure dataset splits so training set has ~66 examples and validation set has ~16 examples — enough for MIPROv2 to produce differentiated scores. The 14 existing hand-labeled examples are reserved for the test set.

**Diagnostic validation (pre-execution gate)**
- R0. Before the full run, validate that dataset size is the root cause of the flat 66.67% scores by running a micro-MIPROv2 trial and a metric sensitivity test.

**Optimization**
- R6. Update `config.yaml` to reference the merged dataset and appropriate split ratios.
- R7. Run `prompt-ops migrate` and verify that trial scores are not all identical AND the optimized prompt outperforms v1.7 baseline on the test set.
- R8. Save the optimized prompt artifact and confirm it preserves v1.7's core methodology.

### Scope Boundaries

**Deferred to Follow-Up Work**
- Fine-tuning optimization hyperparameters (num_candidates, max_bootstrapped_demos) — use defaults for this run
- A/B testing the optimized prompt against v1.7 in the test-agent pipeline
- Manual review and editing of the optimized prompt text

**Outside this product's identity**
- Modifying the v1.7 prompt structure, output schema, or core methodology
- Fixing prompt-ops / dspy compatibility bugs (those are upstream tooling issues)
- Creating a new prompt version from scratch (v1.8+)

### Dependencies

- DeepSeek API key (available in `.env`)
- Oracle model API key (GPT-4 or Claude) for golden-answer generation — required for the anti-circularity strategy (see KTD1)
- v1.7 system prompt (`system_prompt.txt`) for baseline reference
- v1.7 prompt markdown with few-shot examples: `D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版\prompts\prompt_A轨_v1.7_修复感知版.md` — the few-shot examples and JSON schema in this file are used as the user prompt template for answer generation
- Existing `config.yaml` as configuration baseline
- **Pre-condition**: All 82 test case files (`.txt`) must be placed in the working directory (e.g., `use-cases/parent-child-coach/data/test_cases/`) before running U0/U1. Source paths: first batch at `C:\Users\h\Downloads\7.6 test_cases\test_cases\`, second batch at `C:\Users\h\Downloads\第二批-7.6（60）\第二批-7.6（60）\`.

## Planning Contract

### Key Technical Decisions

- **KTD1. Golden-answer generation strategy (revised for anti-circularity)**. The original plan (use v1.7 prompt to generate answers for the 82 training examples) contains a circular optimization trap: training on v1.7's own outputs means the optimizer can at best match v1.7, not exceed it. The validation scores would measure how well candidates reproduce v1.7, not how well they perform on real quality. Three options were considered:
  - **Option A (RECOMMENDED — pending confirmation)**: Use a stronger model (GPT-4 or Claude) as the oracle for training answer generation. This breaks the circularity by providing a higher-quality target than v1.7 itself can produce. The 82 new examples are generated by the oracle model; the 14 existing hand-labeled examples remain as the independent test set.
  - **Option B**: Use the 14 existing hand-labeled examples for training, and the 82 new examples for validation/test. This is the simplest anti-circularity approach but reduces training data to only 14 examples, which may be insufficient.
  - **Option C**: Hybrid — generate with v1.7, then have a stronger model review and correct a sample. Trades off oracle cost but adds a second pass overhead.
  - **Decision**: Option A is documented as the recommended approach. Final confirmation from user required before U2 implementation. If Option A is rejected, fall back to Option B.
- **KTD2. Dataset split strategy (revised for 96 examples)**. The 14 existing hand-labeled examples are reserved as the test set (highest quality, independent from generation). The 82 new oracle-generated examples are split into training (~66, 69%) and validation (~16, 17%). This gives train=66, val=16, test=14 — enough granularity for meaningful score differentiation (validation scores now have 17 possible values: 0/16 through 16/16, compared to the previous 4 possible values from 0/3 through 3/3).
- **KTD3. Metric unchanged**: Keep `selected_fields_comparison` on `should_popup`, `tone`, `popup_insight` — the metric already captures the core behavioral dimensions. The larger validation set alone should fix the differentiation problem.
- **KTD4. Answer generation**: Use a Python script that calls the oracle model API with the full v1.7 prompt (system_prompt + user_prompt with few-shot examples and JSON schema) to generate answers. Output format must match the existing dataset.json schema exactly.

### Assumptions

- The 82 test case files (covering 22 scenario categories with 2-6 variants each across 5 difficulty levels) provide sufficient diversity for the optimizer to learn generalizable patterns
- The oracle model (GPT-4 or Claude) generates higher-quality answers than v1.7, serving as a valid optimization target
- The existing 14 hand-labeled examples have accurate golden answers suitable for independent test evaluation
- DeepSeek API and oracle model API rate limits will not block the answer-generation pass (82 sequential calls)
- The flat 66.67% scores from the previous run are caused by small dataset size, not by a metric or DSPy configuration issue (this is validated in U0 before the full run)

## Implementation Units

### U0. Stage 0 — Diagnostic validation (pre-execution gate)

- **Goal**: Before committing to a full 82-answer generation and optimization run, empirically validate that dataset size is the root cause of the flat 66.67% scores — not a metric sensitivity issue or DSPy configuration problem.
- **Requirements**: R0
- **Dependencies**: None
- **Files**: (temporary artifacts — results discarded after validation)
- **Approach**:
  1. Select 3-5 test cases stratified by difficulty (one each from A, C, D, E levels)
  2. Generate oracle answers for these cases using the oracle model
  3. Run a micro-MIPROv2 trial with a small train/val split (e.g., 3 train / 2 val)
  4. Test metric sensitivity by injecting a deliberately degraded prompt (e.g., remove the 七层结构 behavioral rules) and confirming that scores drop measurably
  5. Gate: only proceed to U1 if the micro-trial shows differentiated scores AND the degraded prompt scores lower
- **Test scenarios**:
  - Positive gate: Small trial shows >=2 distinct score values; degraded prompt scores meaningfully lower than baseline
  - Negative gate 1: Scores remain identical even with 5 cases and degraded prompt → root cause is NOT dataset size → investigate metric or DSPy configuration before proceeding
  - Negative gate 2: Degraded prompt scores the same as good prompt → metric is insensitive → fix metric before proceeding
- **Verification**: Micro-trial log shows score differentiation; degraded prompt score < baseline prompt score

### U1. Convert test case files to structured questions

- **Goal**: Parse all 82 test case `.txt` files, strip metadata headers, and produce a clean JSON array of `{"question": "<dialogue text>"}` entries ready for answer generation.
- **Requirements**: R1, R2
- **Dependencies**: U0 (gate passed)
- **Files**:
  - `use-cases/parent-child-coach/scripts/convert_test_cases.py` (create)
  - `use-cases/parent-child-coach/data/test_cases_questions.json` (create — intermediate output)
- **Approach**: Walk the `test_cases/` directory tree, read each `.txt` file, split on the first blank line to separate the metadata header from dialogue content, trim whitespace, and collect into a JSON array. Preserve the category/subcategory from the header as optional metadata for analysis. Both batches share the same format — a single conversion script handles all 82 files.
- **Patterns to follow**: Existing `dataset.json` question field format — newline-separated dialogue lines with speaker prefixes.
- **Test scenarios**:
  - Happy path: Run script on the 82-file directory → produces JSON array with 82 entries, each having non-empty `question` field
  - Edge case: Files with only metadata header and no dialogue → skipped with warning
  - Edge case: Dialogue lines with leading/trailing whitespace → trimmed correctly
  - Edge case: Second batch files with `_N` suffix variants → correctly parsed (same format as first batch)
- **Verification**: `jq 'length' data/test_cases_questions.json` returns 82; spot-check 5 files across different categories and batches for correct header stripping.

### U2. Generate golden answers using oracle model

- **Goal**: For each of the 82 test case questions, call the oracle model API (GPT-4 or Claude, per KTD1) with the full v1.7 prompt as context to generate a golden answer JSON, producing a complete dataset split file.
- **Requirements**: R3
- **Dependencies**: U1
- **Files**:
  - `use-cases/parent-child-coach/scripts/generate_answers.py` (create)
  - `use-cases/parent-child-coach/data/test_cases_with_answers.json` (create)
- **Approach**: Load the v1.7 system prompt and the full user prompt (including few-shot examples and JSON schema from `D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版\prompts\prompt_A轨_v1.7_修复感知版.md`). For each question, send a chat completion request to the oracle model. Parse the response to extract valid JSON with `should_popup`, `tone`, `popup_insight`, `popup_suggestion` fields. Log failures for manual review. Add rate-limit handling (1s delay between calls).
- **Patterns to follow**: Existing `dataset.json` answer field format — JSON string with the four output fields.
- **Execution note**: This is a data-preparation step; verify a random sample of 5-8 generated answers for quality before proceeding to U3. **KTD1 decision must be confirmed before this step** — if Option A is rejected, fall back to Option B and skip this step for training examples.
- **Test scenarios**:
  - Happy path: All 82 questions receive valid JSON answers with required fields
  - Error path: API call fails for a question → logged, script continues to next question
  - Edge case: API returns valid JSON but missing `popup_suggestion` field → filled with empty string
- **Verification**: `jq 'length' data/test_cases_with_answers.json` returns 82; `jq '.[] | select(.answer | fromjson | .should_popup == null)'` returns empty.

### U3. Merge datasets and configure splits

- **Goal**: Combine the 82 new examples (training + validation pool) with the 14 existing examples (test set) into a single dataset.json, replacing the old 14-example file.
- **Requirements**: R4, R5
- **Dependencies**: U2
- **Files**:
  - `use-cases/parent-child-coach/scripts/merge_dataset.py` (create)
  - `use-cases/parent-child-coach/dataset.json` (overwrite)
- **Approach**: Load existing 14 examples and new 82 examples. Check for duplicate `question` text between the old and new sets; log a warning if any are found (expected: none, since the 82 new cases are user-provided and distinct from the 14 existing examples). Write a merged `dataset.json` with all ~96 examples. The split (train/val/test) is handled by prompt-ops at runtime via config.yaml settings — no need to pre-split. **Note**: prompt-ops `load_dataset()` uses sequential splitting (not random shuffling), so the order of examples in `dataset.json` matters. To ensure diverse representation in each split, the merge script should interleave examples from different difficulty categories rather than grouping all A-level cases together. Update `config.yaml` to set appropriate split ratios.
- **Patterns to follow**: Existing `dataset.json` schema — array of `{"question": "...", "answer": "..."}`.
- **Test scenarios**:
  - Happy path: Merged dataset has exactly 96 entries; all have both `question` and `answer` fields
  - Edge case: Verify no duplicate questions between old and new sets
  - Edge case: Confirm interleaving strategy produces balanced category distribution across the first N entries
- **Verification**: `jq 'length' dataset.json` returns 96; `jq '.[] | select(.answer == null or .question == null)' dataset.json` returns empty.

### U4. Update optimization configuration

- **Goal**: Update `config.yaml` with the merged dataset path, appropriate split configuration, and any learned parameter adjustments.
- **Requirements**: R6
- **Dependencies**: U3
- **Files**:
  - `use-cases/parent-child-coach/config.yaml` (modify)
- **Approach**: Set `dataset.path` to the merged file. Configure explicit `train_size` and `val_size` ratios: `train_ratio: 0.69, val_ratio: 0.17, test_ratio: 0.14` — which yields approximately train=66, val=16, test=14 for the ~96-example dataset. Keep model, metric, and optimization strategy settings unchanged from the last working run. Document the split rationale in comments.
- **Patterns to follow**: Existing `config.yaml` structure.
- **Test scenarios**:
  - Happy path: `config.yaml` parses correctly (valid YAML); dataset path points to existing file
  - Edge case: Verify split ratios produce at least 15 validation and 12 test examples
- **Verification**: `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` succeeds; spot-check key fields; confirm ratios sum to 1.0 (0.69 + 0.17 + 0.14 = 1.00).

### U5. Run optimization and validate results

- **Goal**: Execute `prompt-ops migrate` with the extended dataset and confirm that MIPROv2 produces differentiated trial scores, the optimized prompt outperforms v1.7 baseline on the test set, and the output is a meaningfully optimized prompt.
- **Requirements**: R7, R8
- **Dependencies**: U4
- **Files**:
  - `use-cases/parent-child-coach/results/config_*.json` (create)
  - `use-cases/parent-child-coach/results/config_*.yaml` (create)
- **Approach**: Run `prompt-ops migrate --config config.yaml --dotenv-path .env --log-level info` from the `use-cases/parent-child-coach/` directory. Capture full output. After completion, verify: (a) trial scores are not all identical, (b) best score > baseline v1.7 score on test set, (c) optimized prompt preserves v1.7 core methodology (spot-check for key terms: 三维六格, 七层结构, 盲区二分).
- **Patterns to follow**: Previous successful run's command line and output expectations.
- **Test scenarios**:
  - Happy path: Optimization completes, trial scores show variance, best program saved
  - Quality gate: At least 3 distinct score values across trials (proves differentiation)
  - Baseline gate: Optimized prompt's test-set score exceeds v1.7 baseline score on the 14 hand-labeled examples
  - Content gate: Optimized prompt contains "三维六格", "盲区二分", "七层结构"
- **Verification**: Optimization exits successfully; `results/` contains new output files; trial log shows non-uniform scores; optimized prompt text passes keyword spot-check; optimized prompt test-set score > v1.7 baseline test-set score.

### U6. Cleanup and finalize

- **Goal**: Remove intermediate data files, archive conversion scripts, and ensure the working directory is clean for future runs.
- **Requirements**: None (hygiene)
- **Dependencies**: U5
- **Files**:
  - `use-cases/parent-child-coach/data/test_cases_questions.json` (delete)
  - `use-cases/parent-child-coach/data/test_cases_with_answers.json` (delete)
  - `use-cases/parent-child-coach/scripts/convert_test_cases.py` (archive)
  - `use-cases/parent-child-coach/scripts/generate_answers.py` (archive)
  - `use-cases/parent-child-coach/scripts/merge_dataset.py` (archive)
- **Approach**: Delete intermediate JSON files (keeping only `dataset.json`). Move one-time-use scripts to a `scripts/archive/` subdirectory to keep the scripts folder clean while preserving them for traceability. Do NOT delete `dataset.json`, `config.yaml`, or any files in `results/`.
- **Test scenarios**:
  - Happy path: `data/` contains only `dataset.json`; `scripts/archive/` contains the three conversion scripts
- **Verification**: `ls data/` shows only `dataset.json`; `ls scripts/archive/` shows `convert_test_cases.py`, `generate_answers.py`, `merge_dataset.py`.

## Verification Contract

- **Data integrity**: `python -c "import json; d=json.load(open('dataset.json')); assert len(d)==96, f'Expected 96, got {len(d)}'"`
- **Config validity**: `python -c "import yaml; yaml.safe_load(open('config.yaml')); print('OK')"`
- **Optimization success gate**: Post-run, check that the output log contains at least 3 distinct score values (proves MIPROv2 differentiated between configurations)
- **Baseline outperformance gate**: Optimized prompt's test-set score (on the 14 hand-labeled examples) exceeds the v1.7 baseline score on the same test set
- **Prompt quality gate**: Optimized prompt text contains all three core methodology terms: `三维六格`, `盲区二分`, `七层结构`
- **Diagnostic gate (U0)**: Micro-trial shows >=2 distinct scores; degraded prompt scores lower than baseline
- **Run command**: `cd use-cases/parent-child-coach && prompt-ops migrate --config config.yaml --dotenv-path .env --log-level info 2>&1 | tee run_log.txt`

## Definition of Done

- [ ] U0 diagnostic validation passes (score differentiation confirmed; degraded prompt scores lower)
- [ ] KTD1 oracle model choice confirmed by user (Option A, B, or C)
- [ ] All 82 test case files converted to structured JSON (U1)
- [ ] Golden answers generated for all 82 test cases using oracle model (U2)
- [ ] Merged dataset of 96 examples written to `dataset.json` (U3)
- [ ] Dataset entries interleaved by difficulty category to avoid sequential bias (U3)
- [ ] `config.yaml` updated with merged dataset and proper split config (0.69/0.17/0.14 ratios) (U4)
- [ ] `prompt-ops migrate` completes successfully (U5)
- [ ] Trial scores show meaningful variance (>=3 distinct values) (U5)
- [ ] Optimized prompt outperforms v1.7 baseline on the 14-example test set (U5)
- [ ] Optimized prompt artifact saved to `results/` (U5)
- [ ] Optimized prompt preserves v1.7 core methodology keywords (U5)
- [ ] Intermediate files cleaned up, scripts archived (U6)

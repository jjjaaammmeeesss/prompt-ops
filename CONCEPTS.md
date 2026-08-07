# CONCEPTS

Project-specific vocabulary for prompt-ops. Each entry teaches its concept to a reader with no prior context.

## Optimization

### MIPROv2
DSPy's Bayesian prompt optimizer. Uses bootstrapped few-shot examples, instruction proposal candidates, and hyperparameter search to find prompts that maximize a metric score. The optimizer treats the prompt as a tunable parameter and the metric as the objective function.

### Metric (DSPy)
A scoring function with the signature `(gold, pred) -> float` in `[0, 1]`. Consumed by MIPROv2 to evaluate each candidate prompt's output quality. The metric defines what "better" means — the optimizer finds prompts that produce higher scores. Not to be confused with software performance metrics or monitoring metrics.

### LLM-as-Judge Metric
A custom DSPy metric that delegates quality evaluation to an LLM. Instead of comparing output strings or fields against a reference, the judge LLM scores the prediction on domain-specific dimensions (tone, insight quality, actionability, etc.) and returns a weighted average. Encodes domain expertise into a rubric once, then applies it consistently across every candidate.

### Weight Redistribution (N/A dimensions)
A scoring mechanism where dimensions marked "not applicable" by the judge have their weight proportionally redistributed to the remaining active dimensions. Prevents the optimizer from gaming the metric by always producing output that satisfies a dimension that should sometimes be absent (e.g., always giving advice when sometimes the correct move is to stay diagnostic).

### File-path Class Loader
prompt-ops mechanism for loading custom metric classes from arbitrary Python file paths. Enables use-case-specific metrics to live alongside their config without being installed as packages. Requires the metric class to inherit from the framework's metric base class.

## Multi-Agent Orchestration

### Master Agent
The tone classifier in the multi-agent pipeline. Decides whether a conversation window should trigger a diagnostic or empowering popup. Inputs perception context and conversation text; outputs a binary tone label. Does not know the correct answer at runtime — it is guessing. Not to be confused with a Teacher (which would possess ground truth).

### Production Agent
The popup text generator. Receives the tone label from the Master and generates an ~80-character popup matching that tone. Executes instructions; does not evaluate or learn. When the Master's tone label is correct but the Production wording is inconsistent, routing Validator feedback back to the Master (rather than Production) creates a confounding signal.

### Validator Agent
A consistency checker, not a knowledge source. Compares the generated popup text against the assigned tone label and outputs a binary `is_consistent` flag. Critically, the Validator can say "this popup does not match the tone label" but cannot say "the correct tone is X." Treating the Validator as a Teacher (who would know the correct answer) is the category error documented in `docs/solutions/architecture-patterns/validator-is-not-a-teacher.md`.

### Blind Flip
A retry strategy in the Teacher-Student pipeline: when the Validator rejects a popup as inconsistent, flip the tone label to the opposite value without re-invoking the Master. Contrasts with the "genuine feedback loop" (re-invoke Master with Validator context), which performed worse due to signal deformation. The blind flip is an exhaustive strategy for binary classification — try A, if rejected try B — not a learning mechanism.

### Signal Deformation
The phenomenon where a Validator's feedback about *production quality* ("the wording does not match the tone label") is misinterpreted by the Master as feedback about *classification accuracy* ("my tone judgment was wrong"). This deformation causes the Master to change its classification behavior in response to production-quality issues, accumulating bias with each rejection. Named and analyzed in the Teacher-Student architecture post-mortem.

## Flagged ambiguities

_None yet._

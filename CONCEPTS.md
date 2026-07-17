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

## Flagged ambiguities

_None yet._

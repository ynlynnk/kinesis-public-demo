# Model 2 evaluation

Model 2 is an evidence-retrieval and explanation layer, so it does not
have CNN-style classification accuracy. Its evaluation is separated
into retrieval planning, guardrails and live grounded-response checks.

| Evaluation | Result | Meaning |
| --- | ---: | --- |
| Query-planner benchmark | 4/4 (100.0%) | Correct scope, intent and tool selection |
| Deterministic guardrails | 8/8 (100.0%) | Provider isolation, fallback, role and scope checks |
| Live grounded-response assertions | Not run | Use `run_model_evaluation.ps1 -LiveModel2` |

## Interpretation

- Planner and guardrail checks are software verification, not language-model accuracy.
- The live grounding score checks factual alignment and safety against authorised session evidence.
- Review the raw responses in `model2_evaluation.json` before quoting the live score.
- A larger expert-labelled question set is required for a research-grade evaluation.

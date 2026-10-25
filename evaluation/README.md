# KINESIS model evaluation

This folder keeps evaluation separate from the public dashboard.

## Run without using the external AI service

From the VS Code PowerShell terminal:

```powershell
cd <path-to-kinesis-public-demo>
.\evaluation\run_model_evaluation.ps1
```

This exports the four Model 1 checkpoint metrics and runs the Model 2 query
planner and guardrail tests.

## Include live Gemini responses

```powershell
.\evaluation\run_model_evaluation.ps1 -LiveModel2
```

The live option temporarily reads `gemini-api-key` from Google Secret Manager,
runs three prepared grounded-response cases, and removes the key from the
terminal environment afterward. The key is never written to an evaluation
file.

## Output files

The generated files are saved in `outputs/evaluation/`:

- `model1_cnn_metrics.md` - readable Model 1 results;
- `model1_cnn_metrics.csv` - values suitable for tables or charts;
- `model1_cnn_metrics.json` - full metrics and confusion matrices;
- `model1_accuracy_improvements.md` - current-versus-candidate comparison;
- `forearm_session_metrics.json` - repeated-window forearm movement and
  fatigue decisions, with the original single-window metrics retained;
- `quadriceps_movement_personalized_metrics.json` - the separate,
  after-calibration movement evaluation;
- `model2_evaluation.md` - readable Model 2 benchmark summary; and
- `model2_evaluation.json` - individual checks and live raw responses.

Model 1 accuracy is calculated on held-out public-dataset windows and subjects.
It is not accuracy on KINESIS device recordings. Model 2 does not have a single
classification accuracy because it retrieves and explains evidence; use its
planner, guardrail and grounded-response pass rates instead.

## Candidate Model 1 improvements

The improvement audit keeps two evaluation conditions separate. The
full-repetition quadriceps movement CNN reaches 55.0% accuracy on completely
unseen participants without calibration. After three physiotherapist-labelled
fitting repetitions per class, it reaches 88.8% on the remaining repetitions
from those held-out participants. This second value must always be labelled
**after supervised calibration**.

The quadriceps fatigue feature-sequence CNN reaches 83.7% accuracy and 94.3%
ROC AUC on five held-out participants. Its labels represent early versus late
protocol phase, so the output is a fatigue-like pattern proxy rather than a
fatigue diagnosis or severity score.

Both improved checkpoints remain candidates. They must not replace the live
dashboard metrics until the matching sensor preprocessing is connected and
validated with the final KINESIS hardware.

## Forearm repeated-window evaluation

The forearm session evaluator keeps the deployed checkpoints and held-out
participants unchanged. It averages ten movement-window probabilities for one
movement decision and nine adjacent fatigue-window probabilities for one
sustained-pattern decision. These figures describe session-level decisions;
they must not be relabelled as single-window checkpoint accuracy.

Two additional real-human forearm datasets were also tested as research
candidates. Neither improved participant-independent, single-electrode
generalisation, so neither candidate replaced the deployed checkpoint. This
negative result is retained in the accuracy-improvement audit instead of being
hidden or converted into an inflated dashboard figure.

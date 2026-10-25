# Model 1 accuracy improvement audit

The improved checkpoints are research candidates. They do not replace the
current live dashboard checkpoints until matching preprocessing is connected
and recordings from the final KINESIS device are validated.

| Task | Evaluation condition | Accuracy | Macro F1 | Extra metric |
| --- | --- | ---: | ---: | ---: |
| Forearm movement | Current CNN, single window | 65.8% | 64.8% | - |
| Forearm movement | Same CNN, 10-window session decision | 72.4% | 71.6% | 72.4% balanced accuracy |
| Forearm fatigue | Current CNN, single window | 76.8% | 76.7% | 81.4% ROC AUC |
| Forearm fatigue | Same CNN, 9-window sustained decision | 81.2% | 81.1% | 85.6% ROC AUC |
| Quadriceps movement | Current window model, no calibration | 41.7% | 39.3% | - |
| Quadriceps movement | Full-repetition CNN, no calibration | 55.0% | 53.2% | 54.7% balanced accuracy |
| Quadriceps movement | Full-repetition CNN after 3 guided repetitions per class | 88.8% | 88.6% | 89.5% balanced accuracy |
| Quadriceps fatigue | Current raw-window model | 59.5% | 55.8% | 54.3% ROC AUC |
| Quadriceps fatigue | Five-feature temporal CNN candidate | 83.7% | 83.6% | 94.3% ROC AUC |

## Correct interpretation

- The forearm repeated-window figures use the same deployed checkpoints
  and untouched held-out participants. They describe a session-level
  decision, not a higher single-window checkpoint accuracy.
- Two additional real-human forearm datasets were tested. Their candidate
  models were rejected because sensor and electrode differences did not
  improve participant-independent single-electrode generalisation.
- The 88.8% movement value is an **after-calibration** result. Nine labelled
  fitting repetitions from each held-out patient were not included in that
  patient's 116-repetition evaluation set.
- The 83.7% fatigue value detects early-versus-late protocol phase from
  measured feature sequences. It is not a clinical fatigue diagnosis.
- The unchanged zero-calibration movement result remains visible because
  personalisation and generalisation are different evaluation conditions.
- All values are held-out public-dataset results. They are not clinical
  accuracy and are not accuracy on recordings from final KINESIS hardware.

## Additional real-human dataset trials

- 10-participant electrode-placement dataset:   https://zenodo.org/records/4039550
- 40-participant hand-gesture sEMG dataset:   https://data.mendeley.com/datasets/ckwc76xr2z/2

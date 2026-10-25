# Model 1 CNN evaluation

These values are read directly from the current checkpoint model cards.
They describe held-out **public-dataset windows and participants**, not
clinical accuracy on the final KINESIS device.

| CNN | Accuracy | Macro F1 | Balanced accuracy | ROC AUC | Test windows | Held-out participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forearm movement | 65.8% | 64.8% | 65.8% | Not applicable | 1920 | 8 |
| Forearm fatigue | 76.8% | 76.7% | 76.8% | 81.4% | 1056 | 3 |
| Quadriceps movement | 41.7% | 39.3% | 39.3% | Not applicable | 605 | 5 |
| Quadriceps fatigue | 59.5% | 55.8% | 57.0% | 54.3% | 888 | 1 |

## Per-class recall

- **Forearm movement:** Neutral: 94.8%, Pronation: 52.2%, Supination: 50.5%.
- **Forearm fatigue:** Low-fatigue phase: 69.5%, Fatigue-like phase: 84.1%.
- **Quadriceps movement:** Correct extension: 37.3%, Incomplete extension: 25.3%, Compensatory leg lift: 55.2%.
- **Quadriceps fatigue:** Low-fatigue phase: 34.3%, Fatigue-like phase: 79.7%.

## Interpretation

- The forearm fatigue checkpoint is currently the strongest of the four.
- Forearm movement is moderate and confuses pronation with supination.
- The quadriceps movement and fatigue checkpoints are not strong enough for autonomous or clinical interpretation.
- Final KINESIS accuracy requires labelled recordings from the actual device, electrode placement, exercises and target users.

# Model 1 readiness for Firebase real-time data

Evaluation was refreshed against the current checkpoint hashes on 23 October
2026. The software pipeline is ready for dense Firebase batches, but the
current checkpoints are not yet ready for autonomous or clinical release on
the KINESIS device.

| CNN | Held-out accuracy | Macro F1 | Readiness decision |
| --- | ---: | ---: | --- |
| Forearm movement | 65.8% | 64.8% | Engineering preview; pronation and supination remain frequently confused |
| Forearm fatigue | 76.8% | 76.7% | Strongest current checkpoint; still needs KINESIS-device validation |
| Quadriceps movement | 41.7% | 39.3% | Do not release as a final interpretation |
| Quadriceps fatigue | 59.5% | 55.8% | Do not release as a final interpretation |

The improved quadriceps fatigue research candidate reached 83.7% accuracy and
83.6% macro F1 on held-out public-dataset participants, but it uses a sequence
of five measured features over 64 cycles. It must not replace the live model
until those same five features and cycle boundaries are implemented and tested
on the device stream. The 88.8% quadriceps movement result is available only
after nine physiotherapist-labelled calibration repetitions for each patient;
it is not general zero-calibration accuracy.

The real-time dashboard now routes dense input to the separate trained
movement and fatigue checkpoints. Before either CNN runs, it requires 500 Hz
sEMG cadence, continuous sequence numbers, valid quaternions, a non-flat
signal, a ten-second rest baseline, and a complete post-baseline analysis
window. Low-confidence or close predictions are withheld as `Uncertain`.

The next training milestone is a labelled KINESIS pilot with participant-level
separation. Record the final sensor placement, exercise, reviewer label, and
wearable re-placement sessions. Retrain only after the data contract passes;
then compare the new checkpoint with the current baseline using macro F1,
per-class recall, confusion matrices, and participant-level held-out testing.


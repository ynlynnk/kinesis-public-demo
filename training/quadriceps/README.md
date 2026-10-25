# Quadriceps Model 1 checkpoints

The dashboard contains two separate quadriceps checkpoints. They are selected
when the user chooses **Quadriceps** in Model 1.

## Movement checkpoint

- Source: [KneE-PAD](https://zenodo.org/records/12112951), CC BY 4.0.
- Inputs: affected-side rectus-femoris sEMG at 200 Hz and the matching IMU
  motion window at 50 Hz.
- Classes: correct seated knee extension, incomplete extension, and
  compensatory leg lift.
- Participant split: 20 training, 4 validation, and 5 held-out test
  participants. No participant appears in more than one split.
- Held-out result: 41.7% accuracy and 39.3% macro F1 over 605 windows.

This result is above the 33.3% three-class chance level, but it is weak. It is
adequate for checking that the dashboard and CNN pipeline execute; it is not
evidence of clinical accuracy.

## Fatigue checkpoint

- Source: [Muscle Fatigue in Cyclist](https://huggingface.co/datasets/YominE/Muscle_Fatigue_Cycling),
  CC BY 4.0 as stated by the dataset card.
- Inputs: right rectus-femoris and right vastus-medialis recordings resampled
  to the dashboard's 200 Hz CNN input rate.
- Classes: low-fatigue phase and fatigue-like phase. The source's transition
  and fatigue labels are combined into the fatigue-like class.
- Participant split: participants 1 and 2 for training and participant 3 as
  the held-out test participant.
- Held-out result: 59.5% accuracy, 55.8% macro F1, and 54.3% ROC AUC over 888
  windows.

The fatigue source has only three participants. This checkpoint demonstrates
execution on real public quadriceps sEMG, but it requires broader validation
and must not be treated as a clinical fatigue score.

## Reproduce training

The raw datasets are kept under `data/raw/quadriceps_movement` and
`data/raw/quadriceps_fatigue`. From `KINESIS_PUBLIC_DEMO`, run:

```powershell
python training/quadriceps/train_real_quadriceps_movement_cnn.py
python training/quadriceps/train_real_quadriceps_fatigue_cnn.py
```

The scripts write the deployable checkpoints to `artifacts/models`.

## Accuracy-improvement candidates

Two additional participant-separated candidates are stored under
`artifacts/models/candidates`:

- The full-repetition movement CNN uses one rectus-femoris sEMG channel and
  co-located six-axis IMU data. It improves zero-calibration accuracy from
  41.7% to 55.0%. With three physiotherapist-labelled fitting repetitions per
  class, accuracy on the held-out participants' remaining repetitions is
  88.8% (88.6% macro F1).
- The fatigue feature-sequence CNN uses five measured sEMG features over 64
  cycles. It reaches 83.7% accuracy, 83.6% macro F1 and 94.3% ROC AUC on five
  held-out participants.

The 88.8% movement result is a **personalised after-calibration result**, not
unseen-participant accuracy. The fatigue labels are an early-versus-late
protocol-phase proxy. See `outputs/evaluation/model1_accuracy_improvements.md`
for the complete comparison and deployment requirements.

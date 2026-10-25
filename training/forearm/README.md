# KINESIS forearm-first data plan

## Fixed first use case

The first Model 1 use case is forearm pronation and supination. The user rests
the forearm on a table with the elbow bent and slowly rotates the palm down and
up. The wearable is kept in the same marked position on the upper forearm for
every recording.

This is the most practical first movement because the forearm sEMG measures
muscle activation while the IMU measures the same rotation. It also matches the
public forearm sEMG dataset used by the real-data verification model:

https://doi.org/10.17632/p77jn92bzg.1

That CC BY 4.0 dataset contains 50 participants performing neutral, pronation,
supination and other forearm movements using an eight-electrode Myo armband at
200 Hz. `train_real_forearm_cnn.py` treats each electrode as an independent
one-channel example and keeps participants separate between training,
validation and testing. It supports an honest real-data engineering benchmark,
but it cannot replace recordings from the KINESIS wearable because the sensor,
placement and logging format are different.

The resulting checkpoint predicts only `Neutral`, `Pronation`, or `Supination`.
It must not be presented as fatigue detection or clinical movement correctness.

## Separate real-data fatigue benchmark

Fatigue is handled by a separate CNN head trained with the CC BY 4.0
**Multi-channel Surface EMG Dataset for Fatigue analysis**:

https://doi.org/10.5281/zenodo.5189275

That dataset contains 15 real participants, eight sEMG channels sampled at
200 Hz, and a controlled 120-second upper-limb hold with a 6 kg load. The
training script uses 5–35 seconds as the low-fatigue protocol phase and
90–120 seconds as the fatigue-like protocol phase. Participants are separated
10/2/3 across training, validation and held-out testing.

Run `train_real_fatigue_cnn.py` to regenerate the fatigue checkpoint and its
small held-out dashboard examples. The dashboard reports the CNN probability
of the fatigue-like phase as a **fatigue-pattern score**. It is not a clinical
fatigue-severity score because the source labels describe protocol phase, not
a diagnosis or self-reported severity.

## What must be recorded

Do not repeatedly download a changing Excel file. The firmware should create a
complete recording file when a session or set ends, and the cloud system should
store that file using a unique recording ID.

For training-quality data, record:

- `semg.csv` at the proposed firmware rate of 500 samples per second. Apply the
  documented EMA filter with alpha 0.25 on the device, but preserve the raw
  samples as well as the filtered values whenever storage permits.
- `imu.csv` at 50 samples per second.
- `labels.csv` containing the start and end of every repetition and its human
  labels.

The current trained checkpoints use public 200 Hz sEMG. The dashboard therefore
filters and baseline-normalises the 500 Hz device signal before resampling it to
200 Hz. Changing the firmware to 1,000 Hz later is possible, but the firmware,
preprocessing configuration, and study documentation must all use the same
declared rate. An uploaded row every two seconds is a transport batch, not the
sEMG sampling rate; each batch must contain the dense high-rate samples.

The sEMG file should contain:

```text
recording_id,participant_id,device_time_us,semg_raw,semg_filtered
```

The IMU file should contain:

```text
recording_id,participant_id,device_time_us,quat_w,quat_x,quat_y,quat_z,calib
```

Use the ESP32 device clock for both streams so they can be aligned later. A
Google Sheet may display summaries, but it should not be the raw high-rate data
store.

## Current Model 1 scope versus the proposed final model

The installed movement and fatigue checkpoints are two real-data-trained,
single-channel sEMG CNN heads. They execute genuine learned weights and provide
the current Model 1 proof of operation. The dashboard also monitors the 50 Hz
quaternion stream, but the IMU is not yet an input to those checkpoints.

The proposed final dual-branch model requires synchronized sEMG and IMU windows
from the same KINESIS sessions, with movement-quality and fatigue ground truth.
Only after collecting those paired labelled recordings should an IMU branch and
fusion layer be trained. Until then, do not describe the current checkpoints as
a multimodal fusion CNN. Quadriceps use also needs its own placement-specific
recordings and validation; the present checkpoints cover the forearm only.

## Labels

Use `labels_template.csv` for each repetition. A physiotherapist or trained
reviewer should assign the movement-quality label after watching the movement.
Do not generate labels from the CNN output.

For the first protocol, use these movement labels:

- `correct`
- `incomplete_rotation`
- `elbow_lifted`
- `wrist_compensation`
- `too_fast_or_jerky`

Store the reviewer's exertion rating separately. Do not convert it into a
fatigue label until the project team has chosen and documented one consistent
fatigue-labelling rule.

## Minimum collection sequence

Start with a small engineering pilot before collecting the full study:

1. Record rest for 10 seconds.
2. Record five slow correct repetitions.
3. Record five repetitions for each defined incorrect variation.
4. Repeat the protocol in a second session after replacing the wearable.
5. Have the reviewer label every repetition without seeing the CNN output.
6. Run `validate_recording.py` on the exported files.

The pilot only checks the recording and labelling process. It is not enough to
claim that the model is accurate. The final train, validation and test split
must keep participants separate so the same person does not appear in both
training and testing.

## Why the existing workbook is not training data yet

`Nexus_Testing.xlsx` is useful as a connectivity test. It has the expected
single sEMG value and quaternion columns, but its rows are several seconds
apart and it has no movement or fatigue labels. A time-series CNN needs the
samples within each movement, not occasional uploaded snapshots.


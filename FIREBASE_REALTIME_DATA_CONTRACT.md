# Firebase real-time sensor contract

Firebase should transport complete dense sensor batches; it must not replace
the sensor sampling rate. A normal two-second upload contains 1,000 sEMG
samples at 500 Hz and 100 IMU samples at 50 Hz. The dashboard checks the device
timestamps inside each batch, not the interval between Firebase uploads.

## Required batch shape

```json
{
  "recording_id": "R001",
  "participant_id": "P001",
  "server_time_iso": "2026-10-23T10:00:00+08:00",
  "semg_rate_hz": 500,
  "imu_rate_hz": 50,
  "semg_samples": [
    {"seq": 0, "device_time_us": 0, "raw": 1012, "filtered": 1012.0},
    {"seq": 1, "device_time_us": 2000, "raw": 1014, "filtered": 1012.5}
  ],
  "imu_samples": [
    {
      "device_time_us": 0,
      "quat_w": 1.0,
      "quat_x": 0.0,
      "quat_y": 0.0,
      "quat_z": 0.0,
      "calib": 3
    }
  ]
}
```

`firebase_sensor_adapter.firebase_payload_to_frame()` converts this payload to
the existing dashboard frame. It rejects sparse summaries, incorrect declared
rates, non-increasing device timestamps, and measured cadences outside the
allowed tolerance.

## Recording protocol required before inference

Each new session begins with ten seconds of quiet rest in the final electrode
placement. This baseline must be recorded again after the wearable is removed
or repositioned. Analysis begins only after a further 2.56 seconds of dense
samples are available. The pre-inference gate also checks sequence continuity,
finite values, quaternion norm, sEMG variation, and the full recording length.

## Model-output contract

Movement and fatigue remain separate CNN outputs. A result is shown as
`Uncertain` when the top score is below 60% or its margin over the second score
is below 15%. These are conservative engineering rejection rules, not clinical
thresholds. Current public-dataset checkpoints are marked as engineering
previews until they pass a new participant-separated evaluation using labelled
KINESIS-device recordings.

Model 2 must receive only stored session facts: input-quality status, Model 1
model versions, released or uncertain status, numerical scores, history, and
physiotherapist review. It must not reinterpret raw sEMG or turn a withheld
Model 1 result into a definite conclusion.

## Signal-quality contract

Before inference, the software records a traffic-light quality result:

- `Good`: all blocking checks passed and there are no device warnings;
- `Needs review`: inference may run, but an IMU calibration or similar warning
  must be visible; and
- `Invalid`: missing samples, unstable cadence, invalid quaternions, flat sEMG,
  excessive abrupt sEMG changes, or an incomplete recording block inference.

The detailed result includes packet-loss counts, timing-outlier fraction,
abrupt-sEMG-change fraction, quaternion norm error, and IMU calibration status.
These are engineering data-quality checks, not clinical measurements.

## Repetition-count contract

Physical repetitions and CNN analysis windows are different measurements and
must never share one field. The edge layer may supply:

```json
{
  "available": true,
  "source": "edge_counter",
  "total_repetitions": 10,
  "correct_repetitions": 8,
  "incorrect_repetitions": 2,
  "correct_repetition_percent": 80.0
}
```

If the edge counter has not supplied both counts, all repetition values remain
`null` and the status is `Awaiting edge repetition counter`. The dashboard may
still show a separately labelled CNN movement-window consistency result.

## Completed-session packet

`session_packet.build_session_packet()` creates one stable summary packet when
a session ends. It contains identifiers and timestamps, exercise details,
sampling information, the signal-quality result, the optional edge repetition
summary, the separate CNN window summary, and the original Model 1 outputs.

Until Firebase persistence is connected, the dashboard can download this JSON
or save it under `data/session_packets/<patient>/<session>/session_packet.json`.
The final architecture should continue uploading dense sensor batches during
the session, then store this smaller completed-session packet as the session
manifest.

## Standard Model 2 report

Every exported report uses the same nine sections: Session Information, Signal
Quality, Movement Finding, Fatigue Finding, Historical Comparison,
Physiotherapist Review, Evidence Used, Next Action, and Safety Note. Model 2 may
word the explanations inside those sections, but it must not change the stored
Model 1 values, repetition counts, signal-quality result, or review decision.


---
title: KINESIS repetition checks and fatigue score
topic: rehabilitation
audience: physiotherapist
scope: system documentation for sessions recorded by the KINESIS patient app
---

# KINESIS repetition checks and fatigue score

## How a repetition is judged

The KINESIS patient app measures the joint angle from the motion sensor and the
muscle signal from the surface EMG sensor. Each repetition goes through rest,
lift, hold at the top and lower. A repetition counts only when it passes every
enabled check; otherwise the patient sees the tip of the first failed check.
The exact thresholds used in a session are stored in that session's report
under `criteria`.

## Range of motion checks

- INCOMPLETE_ROM: the peak angle did not reach the target minus a small
  tolerance. Repeated misses can mean the target is above the current comfortable
  range, or reduced effort late in a set.
- OVER_FLEX: the peak went well past the target or the calibrated personal range.
- NOT_RETURNED: the joint was not lowered back to the start before the next lift.

## Tempo and control checks

- EARLY_DOWN: the patient did not stay near the top long enough (a short pause
  is required). Frequent early lowering combined with an overshoot of the target
  usually means the movement is being swung up and dropped rather than controlled.
- SWING_FAST and DROP_FAST: the lift or the lowering was too quick, either in
  duration or in peak angular speed. Fast lowering reduces the eccentric work the
  exercise is meant to train.
- JERKY: low movement smoothness (spectral arc length, SPARC). Smoothness often
  drops when a patient tires or compensates.
- TIMEOUT: a single repetition lasted too long.

## Compensation checks

- FOREARM_LIFT: the rest of the limb moved (rotation off the joint axis or extra
  acceleration), which suggests the movement is being helped by the shoulder or
  trunk.
- WRIST_TWIST and SIDE_BEND: rotation or sideways movement outside the exercise
  plane. Some exercises switch these off.

## Muscle checks

- NO_MUSCLE: the working muscle's activation was low relative to the reference
  contraction recorded during calibration, or it switched on after the movement
  had already started. This pattern fits a lift driven by momentum or by the other
  hand (an assisted curl).
- SIGNAL_POOR: the muscle or motion signal was unreliable during the repetition
  (electrode lift-off, clipping, mains hum or a sensor dropout). Findings from such
  repetitions should not be interpreted.

## The fatigue score

The fatigue score (0-100) compares each counted repetition with the patient's
own baseline, the average of the first counted repetitions of the same session.
It weights the drop in EMG median frequency, the rise of a spectral fatigue
index, the rise in effort (RMS as a percentage of the reference contraction) and
the rise in lift time, then smooths the result over repetitions. Green is below
40, amber 40-70 and red above 70. When effort and median frequency rise together
the effort term is ignored, because that pattern means more force rather than
fatigue. When too few repetitions were counted the score is a demonstration
estimate over all attempts and should not be interpreted clinically.

## Reading a session

Look first at signal quality, then at which check failed most often and whether
failures cluster late in the session (fatigue) or from the start (technique,
target too high, or misunderstanding the instruction). Compare the good-repetition
rate, average peak angle and fatigue score with earlier sessions before drawing
conclusions about progress.

---
title: Understanding kinematic measurements
topic: imu
audience: physiotherapist
scope: general knowledge and KINESIS measurement boundaries
---

# Understanding kinematic measurements

## What kinematics means

Kinematics describes how a body or body segment moves without explaining the forces that caused the movement.

Common kinematic measurements include:

- position;
- displacement;
- joint angle;
- range of motion;
- movement direction;
- linear velocity;
- angular velocity;
- linear acceleration;
- movement duration; and
- repetition timing.

Different measurements require suitable sensor placement, calibration, processing, and validation.

## Accelerometer measurements

An accelerometer measures acceleration along one or more axes.

Its output may be influenced by:

- physical movement;
- gravity;
- sensor orientation;
- vibration;
- impact;
- attachment movement; and
- measurement noise.

Acceleration is not the same as movement distance or speed. Additional processing and assumptions are needed to estimate those quantities.

## Gyroscope measurements

A gyroscope measures angular velocity, meaning how quickly the sensor rotates.

Angular velocity is not the same as an exact joint angle.

Estimating joint angle normally requires:

- known sensor orientation;
- appropriate calibration;
- a suitable sensor-fusion method;
- consistent attachment to the body segment; and
- validation against an accepted reference method.

## Range of motion

Range of motion describes the difference between the minimum and maximum angle reached during a movement.

A valid range-of-motion value generally requires calibrated angle measurements.

A normalised movement graph cannot be converted directly into degrees of joint motion.

## Movement timing

Time-based measurements may include:

- session duration;
- repetition duration;
- time between repetitions;
- timing of movement peaks; and
- timing differences between EMG and IMU activity.

Timing can help describe movement patterns, but timing alone does not determine whether the movement was clinically correct.

## What KINESIS currently provides

The current KINESIS visualisation provides:

- time in seconds;
- normalised EMG activation;
- normalised IMU movement;
- window-level movement probabilities;
- a movement classification; and
- movement confidence.

The normalised IMU movement is created from processed acceleration and angular-velocity information.

## Current KINESIS limitations

Unless new validated processing is added, KINESIS must not claim to provide:

- exact joint angles;
- joint range of motion in degrees;
- movement distance;
- walking speed;
- limb position;
- repetition counts;
- movement smoothness scores;
- balance measurements;
- gait measurements; or
- injury-risk measurements.

The agent must not calculate or invent these values from the normalised graph.

## Comparing sessions

Before comparing kinematic measurements, check whether the sessions used the same:

- exercise;
- sensor location;
- sensor orientation;
- recording duration;
- starting position;
- movement instructions; and
- processing method.

The current KINESIS graph is normalised separately within each session.

Therefore, compare general patterns and Model 1 outputs rather than treating graph height as an absolute physical measurement across sessions.

## Safe interpretation examples

Use:

- "The movement signal increased during this part of the session."
- "The session contains a repeated movement pattern."
- "The timing of the movement pattern differs from the previous session."
- "The displayed signal is normalised and does not represent joint angle."
- "Further validated processing would be needed to calculate range of motion."

Avoid:

- "The patient's knee moved through 90 degrees."
- "The patient moved twice as fast."
- "This peak represents the exact joint position."
- "The graph confirms reduced range of motion."
- "The patient has a balance problem."

## Future development

If exact kinematic measurements are added later, the project should document:

1. The IMU and body location used.
2. The calibration procedure.
3. The coordinate system.
4. The sensor-fusion or measurement algorithm.
5. The exercise protocol.
6. The reference system used for validation.
7. Measurement error and reliability.
8. The populations on which the method was tested.

Until those steps are completed, the system should describe only the measurements it actually produces.

## Important limitation

KINESIS results support professional review and are not a medical diagnosis.

The IMU graph and movement classification support system testing and physiotherapist review. They are not a medical diagnosis or a replacement for validated clinical measurements.

## Sources

1. Leardini A, Lullini G, Giannini S, Berti L, Ortolani M, Caravaggi P. Validation of the angular measurements of an inertial-measurement-unit-based rehabilitation system. Journal of NeuroEngineering and Rehabilitation. 2014. https://pubmed.ncbi.nlm.nih.gov/25212257/

2. Mitternacht J, Hermann A, Carqueville P. Acquisition of lower-limb motion characteristics with a single inertial measurement unit: validation for use in physiotherapy. Diagnostics. 2022. https://pubmed.ncbi.nlm.nih.gov/35885542/

3. Nilsson S, Ertzgaard P, Lundgren M, Grip H. Test-retest reliability of kinematic and temporal outcome measures for clinical gait and stair walking tests based on wearable inertial sensors. Sensors. 2022. https://pubmed.ncbi.nlm.nih.gov/35161916/

4. McGrath T, et al. Evaluating the use of machine learning in the assessment of joint angle using a single inertial sensor. Journal of Rehabilitation and Assistive Technologies Engineering. 2019. https://pubmed.ncbi.nlm.nih.gov/31452927/

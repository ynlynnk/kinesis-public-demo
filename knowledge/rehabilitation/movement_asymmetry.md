---
title: Understanding movement asymmetry
topic: rehabilitation
audience: physiotherapist
scope: general knowledge and KINESIS measurement boundaries
---

# Understanding movement asymmetry

## Plain-language explanation

Movement asymmetry means that corresponding movements on two sides of the body differ in some measurable way.

Examples may include differences between the left and right sides in:

- movement timing;
- movement distance;
- joint angle;
- movement speed;
- step length;
- stance time;
- muscle-activation timing; or
- muscle-activation magnitude.

An observed difference is not automatically a clinical problem. Interpretation depends on the exercise, measurement method, person's condition, and professional assessment.

## Different types of asymmetry

### Temporal asymmetry

Temporal asymmetry describes differences in timing.

Examples include:

- one side taking longer to complete a movement;
- unequal stance or swing times; or
- one side beginning muscle activation earlier.

### Spatial asymmetry

Spatial asymmetry describes differences in position or movement distance.

Examples include:

- unequal step lengths;
- different joint ranges;
- different movement paths; or
- unequal displacement.

### Muscle-activation asymmetry

Muscle-activation asymmetry describes differences between corresponding EMG recordings.

These differences may be influenced by physiological activity as well as electrode placement, skin contact, signal quality, and normalisation.

## Requirements for measuring asymmetry

A valid left-versus-right comparison normally requires:

- sensors on both relevant sides;
- clearly identified left and right channels;
- consistent sensor placement;
- suitable calibration;
- synchronised recordings;
- the same movement task;
- an appropriate asymmetry calculation; and
- validation of that calculation.

Without these requirements, the system should not claim to have measured asymmetry.

## Current KINESIS limitation

The current KINESIS session output does not provide a validated left-versus-right asymmetry measurement.

It provides:

- a combined normalised EMG activation pattern;
- a combined normalised IMU movement pattern;
- a fatigue classification;
- a movement classification; and
- model-confidence information.

Therefore, the agent must not claim that KINESIS detected:

- left-side weakness;
- right-side weakness;
- unequal joint angles;
- gait asymmetry;
- unequal range of motion; or
- compensation by a specific limb.

The current movement classification may indicate that the session's pattern differs from the pattern learned by Model 1. It does not identify a specific side or prove that asymmetry caused the classification.

## Reviewing a possible movement difference

If future KINESIS versions include validated bilateral measurements, the physiotherapist should consider:

1. Which sides and body segments were measured.
2. Whether the same sensors and placement were used.
3. Whether recordings were synchronised.
4. Whether the exercise was performed under comparable conditions.
5. Whether the difference was consistent across repetitions.
6. Whether the difference exceeded expected measurement variation.
7. Whether the result agrees with professional observation.
8. Whether additional assessment is required.

## Comparing sessions

Do not describe a changed movement-confidence score as reduced or increased asymmetry.

The confidence score refers to the model's classification confidence. It is not an asymmetry index.

A valid longitudinal asymmetry comparison requires the same measurement definition and comparable recording conditions across sessions.

## Safe explanation examples

For KINESIS, use:

- "The model detected a movement pattern that may need physiotherapist review."
- "The current data does not identify whether the difference occurred on the left or right side."
- "KINESIS does not currently calculate a validated asymmetry measurement."
- "Bilateral sensor measurements would be required for a left-versus-right comparison."

If validated asymmetry measurements are added later, use:

- "A difference was measured between the two sides."
- "The meaning of the difference requires physiotherapist interpretation."
- "The difference was observed across several comparable repetitions."

Avoid:

- "The patient's left side is weak."
- "The patient is compensating with the right leg."
- "The movement-confidence percentage is an asymmetry score."
- "The graph confirms abnormal asymmetry."
- "Asymmetry proves that rehabilitation has worsened."

## Important limitation

KINESIS does not currently provide a validated bilateral asymmetry measurement.

This information is general background only. It is not a diagnosis and does not replace the physiotherapist's assessment.

## Sources

1. Buckley C, Mico-Amigo ME, Dunne-Willows M, et al. Gait asymmetry post-stroke: determining valid and reliable methods using a single accelerometer located on the trunk. Sensors. 2019. https://pubmed.ncbi.nlm.nih.gov/31861630/

2. Lanotte F, Shin SY, O'Brien MK, Jayaraman A. Validity and reliability of a commercial wearable sensor system for measuring spatiotemporal gait parameters in a post-stroke population. Physiological Measurement. 2023. https://pubmed.ncbi.nlm.nih.gov/37557187/

3. Desai R, Martelli D, Alomar JA, Agrawal S, Quinn L, Bishop L. Validity and reliability of inertial measurement units for gait assessment within a post-stroke population. Topics in Stroke Rehabilitation. 2023. https://pubmed.ncbi.nlm.nih.gov/37545107/

4. Felius RAW, et al. Reliability of IMU-based gait assessment in clinical stroke rehabilitation. Sensors. 2022. https://pubmed.ncbi.nlm.nih.gov/35161654/

---
title: Understanding IMU movement analysis
topic: imu
audience: physiotherapist
scope: general knowledge and KINESIS movement interpretation
---

# Understanding IMU movement analysis

## What an IMU measures

An inertial measurement unit, or IMU, is a wearable sensor that can record aspects of physical movement.

Common IMU measurements include:

- acceleration;
- angular velocity; and
- changes in sensor orientation.

IMU data can help describe movement timing, repetition, speed, smoothness, and consistency when the sensor and analysis method are appropriate.

## What the KINESIS movement graph shows

KINESIS combines processed acceleration and angular-velocity information into a normalised movement signal.

The displayed movement values range from 0 to 1 within the selected session:

- values closer to 1 represent relatively greater movement activity within that recording;
- values closer to 0 represent relatively lower movement activity within that recording.

The graph does not display an exact joint angle, movement distance, walking speed, or clinical range of motion.

Because each session is normalised separately, values should not be treated as absolute measurements across different sessions.

## KINESIS movement classification

Model 1 produces:

- a movement result;
- a movement-confidence score; and
- window-level probabilities for an incorrect movement pattern.

The movement result has two internal labels:

- `correct`
- `incorrect`

Display these as:

- "Movement matched the learned pattern"
- "Movement may need correction"

Avoid presenting `correct` as proof of perfect exercise technique.

Avoid presenting `incorrect` as a diagnosis or definite clinical error. The result means that the model detected a pattern associated with the labelled examples used during training.

## Movement confidence

Movement confidence describes how strongly Model 1 supports its final movement classification.

It does not measure:

- the severity of a movement problem;
- the risk of injury;
- the patient's rehabilitation progress;
- the percentage of the exercise completed correctly; or
- the amount of correction required.

For example, 90% movement confidence does not mean that 90% of the exercise was performed correctly.

## Window-level movement probability

The window-level value represents the model's estimated probability of an incorrect movement pattern during that analysis window.

Use the sequence of windows to identify:

- isolated changes;
- repeated changes;
- sustained model findings; and
- when a pattern occurred during the session.

Do not treat one isolated window as sufficient evidence for a clinical conclusion.

## Factors affecting IMU interpretation

IMU measurements may be influenced by:

- sensor placement;
- sensor orientation;
- loose attachment;
- sensor movement relative to the body;
- differences in starting position;
- exercise speed;
- exercise range;
- pauses between repetitions;
- changed exercise instructions; and
- the body segment on which the sensor is worn.

Session comparisons are more meaningful when sensor placement and exercise conditions are consistent.

## Comparing IMU and EMG patterns

The IMU and EMG signals provide different information:

- IMU indicates relative physical movement;
- EMG indicates relative muscle-associated electrical activity.

Viewing them together can help examine timing.

The agent may describe whether the signals increased, decreased, or peaked at similar times. It must not claim that timing alone proves that the movement was correct or incorrect.

## What the physiotherapist should review

Consider:

1. Whether the correct exercise was performed.
2. Whether the sensor was placed and secured consistently.
3. Whether the movement pattern was sustained or isolated.
4. Whether pauses or unusual movements affected the recording.
5. Whether the result agrees with professional observation.
6. Whether previous sessions used comparable conditions.
7. Whether the AI result should be confirmed, corrected, or marked uncertain.
8. Whether a review comment should explain the decision.

## Safe explanation examples

Use:

- "The model found a movement pattern that may need correction."
- "The model found that the movement matched its learned pattern."
- "The movement-confidence score describes confidence in the AI classification."
- "An isolated change appeared during this part of the session."
- "Sensor placement and exercise conditions should be considered."
- "The physiotherapist changed the movement classification after review."

Avoid:

- "The patient performed 90% of the exercise correctly."
- "This graph provides the exact joint angle."
- "The patient has an abnormal movement disorder."
- "The movement will cause an injury."
- "This session proves that rehabilitation is improving."

## Important limitation

KINESIS results support professional review and are not a medical diagnosis.

The movement result supports physiotherapist review and system testing. It is not a medical diagnosis and does not replace direct professional assessment.

## Sources

1. Leardini A, Lullini G, Giannini S, Berti L, Ortolani M, Caravaggi P. Validation of the angular measurements of an inertial-measurement-unit-based rehabilitation system. Journal of NeuroEngineering and Rehabilitation. 2014. https://pubmed.ncbi.nlm.nih.gov/25212257/

2. Okita S, Schwerz De Lucena D, Chan V, Reinkensmeyer DJ. Measuring movement quality of the stroke-impaired upper extremity with a wearable sensor. IEEE Engineering in Medicine and Biology Society Conference. 2021. https://pubmed.ncbi.nlm.nih.gov/34892643/

3. Mitternacht J, Hermann A, Carqueville P. Acquisition of lower-limb motion characteristics with a single inertial measurement unit: validation for use in physiotherapy. Diagnostics. 2022. https://pubmed.ncbi.nlm.nih.gov/35885542/

4. Llamas-Ramos R, et al. Unsupervised IMU-based evaluation of at-home exercise programmes: a feasibility study. BMC Sports Science, Medicine and Rehabilitation. 2022. https://pubmed.ncbi.nlm.nih.gov/35183244/

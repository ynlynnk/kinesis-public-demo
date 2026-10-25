---
title: Interpreting fatigue indicators
topic: fatigue
audience: physiotherapist
scope: general knowledge and KINESIS output interpretation
---

# Interpreting fatigue indicators

## Indicators produced by KINESIS

KINESIS Model 1 analyses the selected sensor recording in consecutive time windows.

It produces:

- a fatigue result;
- a fatigue score;
- an estimated fatigue-onset time; and
- fatigue probabilities for individual time windows.

These are AI outputs and must remain separate from the physiotherapist's review.

## Fatigue result

The fatigue result has two possible labels:

- `non_fatigued`: the model did not find a sustained fatigue pattern;
- `fatigued`: the model found a sustained fatigue pattern.

Display these labels as:

- "No fatigue pattern detected"
- "Fatigue pattern detected"

Do not present the result as a medical diagnosis.

## Fatigue score

The fatigue score summarises the strongest sustained fatigue pattern identified across the session.

A higher score means the model found stronger evidence for the learned fatigue pattern.

The score does not indicate:

- the severity of a medical condition;
- the amount of pain experienced;
- the person's overall tiredness;
- whether treatment was successful; or
- whether a rehabilitation plan should change.

When displayed as a percentage, describe it as the model's fatigue-pattern score. Do not describe it as the percentage severity of fatigue.

## Estimated fatigue onset

If a sustained fatigue pattern is detected, KINESIS reports the beginning of the first group of consecutive time windows that passed the model's fatigue threshold.

Explain it as:

"The model first detected a sustained fatigue pattern at approximately [time] seconds."

The reported time is an estimate based on analysis windows. It is not the exact physiological moment when fatigue began.

If no onset is available, display:

"No sustained fatigue onset was detected."

## Window-level probabilities

Window-level probabilities show how the model's output changed during the recording.

Use them to:

- show the pattern over time;
- identify where sustained model evidence first appeared; and
- support the session visualisation.

Do not diagnose fatigue from a single isolated high-value window. Interpret it together with neighbouring windows, signal quality, session conditions, and the physiotherapist's assessment.

## Physiological context

Surface EMG characteristics can change during a fatiguing muscular task. Research has examined changes in signal frequency and amplitude as possible fatigue indicators.

However, these indicators may vary according to:

- the muscle being measured;
- the type and intensity of contraction;
- electrode placement;
- signal quality;
- exercise duration;
- the individual; and
- the experimental protocol.

Therefore, one EMG indicator or one model output is not sufficient for a clinical conclusion.

## What the physiotherapist should review

When reviewing a KINESIS fatigue result, consider:

1. Whether the correct patient and session were selected.
2. Whether the recording appears complete.
3. Whether the exercise and recording conditions were appropriate.
4. Whether the fatigue pattern is sustained rather than isolated.
5. Whether the estimated onset agrees with the session evidence.
6. How the result compares with previous similar sessions.
7. Whether the AI result should be confirmed, corrected, or marked uncertain.
8. Whether an explanatory review comment should be recorded.

## Safe summary examples

Use:

- "The model did not detect a sustained fatigue pattern."
- "The model detected a sustained fatigue pattern."
- "The fatigue-pattern score was 72%."
- "The model first detected the pattern at approximately 18 seconds."
- "The physiotherapist changed the model's fatigue result."
- "More information is required before confirming this result."

Avoid:

- "The patient is 72% fatigued."
- "The patient definitely developed fatigue at 18 seconds."
- "The result proves that the patient has recovered."
- "The rehabilitation programme must be changed."

## Important limitation

KINESIS results support professional review and are not a medical diagnosis.

The fatigue indicators are intended to support review and system testing. They are not a diagnosis and do not replace the physiotherapist's professional judgement.

## Sources

1. Allison GT, Fujiwara T. The relationship between EMG median frequency and low frequency band amplitude changes at different levels of muscle capacity. Clinical Biomechanics. 2002. https://pubmed.ncbi.nlm.nih.gov/12135548/

2. Dolan P, Mannion AF, Adams MA. Fatigue of the erector spinae muscles: a quantitative assessment using frequency banding of the surface electromyography signal. Spine. 1995. https://pubmed.ncbi.nlm.nih.gov/7716619/

3. Krivickas LS, Taylor A, Maniar E, Mascha E, Reisman SS. Is spectral analysis of the surface electromyographic signal a clinically useful tool for evaluation of skeletal muscle fatigue? Journal of Clinical Neurophysiology. 1998. https://pubmed.ncbi.nlm.nih.gov/9563580/

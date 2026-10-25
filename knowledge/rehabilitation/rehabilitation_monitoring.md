---
title: Rehabilitation session monitoring
topic: rehabilitation
audience: physiotherapist
scope: general knowledge and KINESIS longitudinal comparison
---

# Rehabilitation session monitoring

## Purpose of session monitoring

Rehabilitation monitoring involves reviewing information collected across multiple sessions to identify patterns, changes, and issues that may require professional attention.

Sensor and AI results can support this process, but they should complement rather than replace:

- physiotherapist assessment;
- patient-reported experience;
- observation of exercise performance;
- validated clinical outcome measures; and
- relevant session context.

## KINESIS longitudinal workflow

When a physiotherapist opens a selected session, the KINESIS agent should:

1. Confirm that the patient and session are authorised.
2. Retrieve the selected session's original Model 1 output.
3. Retrieve previous authorised sessions for the same patient.
4. Compare the available session results.
5. Retrieve the physiotherapist review status and history.
6. Present the comparison in plain language.
7. Keep the AI result separate from the human-reviewed decision.
8. State limitations and missing information.

## Information currently available for comparison

KINESIS can compare:

- session ID;
- fatigue classification;
- fatigue score;
- estimated fatigue-onset time;
- movement classification;
- movement confidence; and
- physiotherapist review status.

When detailed review history is available, it may also show:

- original AI labels;
- final reviewed labels;
- reviewer ID;
- review date and time;
- review comments;
- review ID; and
- permission for possible future AI improvement.

## Correct interpretation of changes

A change means that two session outputs are different.

A change does not automatically prove improvement or deterioration.

Examples:

- A lower fatigue score means the model found less evidence for its learned fatigue pattern in that recording.
- A later fatigue-onset estimate means the model first detected the pattern later in that recording.
- A changed movement label means Model 1 classified the session differently.
- A higher movement-confidence score means the model was more confident in its selected label.

Movement confidence must always be interpreted together with the movement label.

For example, high confidence in `incorrect` is not an improvement over lower confidence in `correct`.

## Conditions for meaningful comparison

Before comparing sessions, consider whether they used similar:

- exercises;
- recording duration;
- movement instructions;
- sensor locations;
- sensor orientation;
- signal-processing methods;
- model version; and
- session conditions.

If these conditions are unknown or different, describe the comparison cautiously.

## Comparing fatigue results

The summary may describe:

- whether the fatigue label changed;
- whether the fatigue score increased or decreased;
- whether onset was detected;
- whether detected onset occurred earlier or later; and
- whether the physiotherapist confirmed or corrected the result.

Use:

- "The fatigue-pattern score was higher than in the previous recorded session."
- "The model detected a fatigue pattern earlier in this session."
- "No onset was detected in either session."
- "The physiotherapist corrected the AI fatigue result."

Do not conclude that the patient's endurance improved or worsened from these values alone.

## Comparing movement results

The summary may describe:

- whether the movement label changed;
- whether confidence in the selected label changed;
- whether the window-level pattern was isolated or sustained; and
- whether the physiotherapist confirmed or corrected the result.

Use:

- "The model classified this session differently from the previous session."
- "Confidence in the current movement classification was higher."
- "The physiotherapist changed the movement result after review."

Do not interpret movement confidence as an exercise-performance score.

## Human-review information

The most recent physiotherapist review should be clearly distinguished from the original AI output.

Use the final reviewed label when describing the professional decision, while retaining the original AI result for traceability.

A review marked for possible future AI improvement:

- does not immediately retrain the model;
- does not modify the original Model 1 output;
- does not guarantee that the review will be used; and
- must be evaluated during a separate controlled model-update process.

## Missing or uncertain information

If information is missing:

- state which information is unavailable;
- continue with the evidence that is available;
- do not estimate or invent a value; and
- avoid drawing a conclusion that depends on the missing information.

If there are not enough previous sessions, say:

"Not enough comparable session history is available to identify a trend."

If session conditions are unknown, say:

"The recorded difference may be influenced by differences in session or sensor conditions."

## User-friendly summary structure

Present session monitoring information in this order:

### Current session

Show the selected patient, session, AI fatigue result, AI movement result, and relevant scores.

### Comparison with previous sessions

Describe only the most important observed differences.

### Physiotherapist review

Show whether the result is awaiting review, confirmed, corrected, or uncertain. Include comments and final reviewed labels when available.

### What this means

Provide a cautious plain-language explanation. Do not diagnose or recommend treatment changes.

### Limitations

Keep the input source visible, state that the output is not a diagnosis, and leave professional interpretation with the physiotherapist.

## Safe explanation examples

Use:

- "The model found less evidence of its learned fatigue pattern than in the previous session."
- "The movement classification changed between these sessions."
- "This difference should be interpreted together with the session conditions."
- "The physiotherapist corrected the AI result."
- "More comparable sessions are needed before describing a trend."

Avoid:

- "The patient is recovering."
- "The patient's condition has worsened."
- "The treatment is working."
- "The exercise programme should be changed."
- "The AI has learned from this review already."
- "This single change is clinically significant."

## Important limitation

KINESIS results support professional review and are not a medical diagnosis.

Its longitudinal comparison identifies differences in model outputs. It does not establish clinical improvement, deterioration, treatment effectiveness, or diagnosis.

Final interpretation remains with the physiotherapist.

## Sources

1. van Meulen FB, et al. Inertial sensor measurements of upper-limb kinematics in stroke patients in clinic and home environment. Frontiers in Bioengineering and Biotechnology. 2018. https://pubmed.ncbi.nlm.nih.gov/29707537/

2. Seo NJ, et al. Wearable sensor to monitor quality of upper-limb task practice for stroke survivors at home. Sensors. 2024. https://pubmed.ncbi.nlm.nih.gov/38257646/

3. Lee SI, et al. Predicting and monitoring upper-limb rehabilitation outcomes using clinical and wearable sensor data in brain injury survivors. IEEE Transactions on Biomedical Engineering. 2021. https://pubmed.ncbi.nlm.nih.gov/32997621/

4. Strohrmann C, Labruyere R, Gerber CN, van Hedel HJA, Arnrich B, Troster G. Monitoring motor capacity changes of children during rehabilitation using body-worn sensors. Journal of NeuroEngineering and Rehabilitation. 2013. https://pubmed.ncbi.nlm.nih.gov/23899401/

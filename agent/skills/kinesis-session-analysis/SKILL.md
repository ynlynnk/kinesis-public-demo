---
name: kinesis-session-analysis
description: Answer a physiotherapist's question about a selected KINESIS rehabilitation session using only relevant authorised MCP evidence and approved general knowledge.
---

# KINESIS Session Analysis

## Purpose

Create a clear, physiotherapist-friendly answer to a question about a selected rehabilitation session.

Use the question to retrieve only the relevant session evidence. The configured Model 2 language model may explain verified findings, but it must never calculate, replace, or alter a Model 1 result.

## Required input

The workflow must receive:

- `patient_id`
- `session_id`
- `requesting_role`
- `user_query`

The full workflow is available only when `requesting_role` is `physiotherapist`.

Never expose full session evidence, review history, or physiotherapist comments to a patient or unauthorised role.

## Workflow

Follow these steps in order.

### 1. Validate access and session selection

Call:

`list_available_sessions(requesting_role="physiotherapist")`

Confirm that the selected `patient_id` and `session_id` exist in the authorised results.

Stop if:

- the patient or session is not found;
- the identifiers do not match;
- the requesting role is not authorised; or
- the session cannot be accessed.

Explain the problem without guessing or exposing other patient records.

### 2. Plan the evidence request

Identify whether the question concerns:

- fatigue;
- movement;
- progress across sessions;
- physiotherapist review; or
- an overall session summary.

Always retrieve the original Model 1 findings. Retrieve trends, graphs, and review records only when they are relevant to the question. An overall summary may use all available evidence.

### 3. Retrieve the original AI findings

Call:

`get_model1_outputs(patient_id, session_id, requesting_role="physiotherapist")`

Keep the original Model 1 output separate from the physiotherapist's review.

Extract, when available:

- fatigue result;
- fatigue probability;
- estimated fatigue onset;
- movement result; and
- movement confidence.

Do not alter the original model result.

### 4. Retrieve historical trends when relevant

Call:

`compare_session_trends(patient_id, requesting_role="physiotherapist")`

Compare the selected session with previous authorised sessions.

Describe changes in plain language. Clearly state when there are not enough previous sessions for a meaningful comparison.

### 5. Retrieve visualisation data when relevant

Call:

`get_session_visualisation(patient_id, session_id, requesting_role="physiotherapist")`

Use the returned data to support session charts or graphs.

Do not invent missing measurements.

### 6. Retrieve the human-review status when relevant

Call:

`get_feedback_status(patient_id, session_id, requesting_role="physiotherapist")`

If `get_review_history` is available, also call:

`get_review_history(requesting_role="physiotherapist")`

Filter the returned history to the selected patient and session.

Show the following when available:

- original AI fatigue result;
- physiotherapist's fatigue decision;
- original AI movement result;
- physiotherapist's movement decision;
- final reviewed labels;
- reviewer ID;
- review time;
- review comments;
- review ID; and
- whether future AI-improvement use was allowed.

If detailed history is unavailable, show only the confirmed status and explain that the complete review record could not be retrieved.

Do not say that a review has retrained or immediately changed the AI model.

### 7. Add optional general knowledge

If an approved KINESIS `knowledge/` collection and the knowledge-base retriever are available, retrieve only information relevant to explaining the session findings.

General knowledge may help explain:

- muscle fatigue;
- surface EMG;
- movement analysis;
- kinematic measurements;
- movement asymmetry; and
- rehabilitation monitoring.

Keep general knowledge separate from patient-specific evidence.

Cite the source document used. General knowledge must never override the Model 1 result or the physiotherapist's decision.

If knowledge retrieval is unavailable, continue using the MCP session data and state that no additional general explanation was added.

### 8. Generate the answer

Answer the user's question directly before adding supporting detail. Use only the retrieved evidence.

For a physiotherapist, detailed Model 1 scores, trends, graph-ready data, and review evidence may be included when relevant.

For a clinician, use only the minimum-necessary coordination summary and authorised trend evidence.

For a patient, use only the patient-summary tool output. Do not reveal raw Model 1 scores, technical evidence, review comments, or another patient's records.

Present the result using the following structure.

## Session identification

Show the patient ID and session ID.

## Fatigue assessment

Show:

- the AI result in plain language;
- fatigue probability;
- estimated fatigue onset, when detected; and
- the physiotherapist's reviewed decision, when available.

Clearly distinguish between the AI result and human review.

## Movement assessment

Show:

- the AI movement result in plain language;
- movement confidence; and
- the physiotherapist's reviewed decision, when available.

Clearly distinguish between the AI result and human review.

## Changes across sessions

Summarise important changes compared with previous sessions.

Avoid claiming that a numerical change is a clinical improvement or deterioration unless that conclusion is supported by the available evidence.

## Physiotherapist review

Show the review status, reviewer, review date, final decisions, and comments when available.

## General explanation

Provide a short, non-patient-specific explanation from the approved knowledge base when available.

## Important limitations

Always state that the answer is not a medical diagnosis and final interpretation remains with the physiotherapist. Never describe a simulator or training record as a live patient recording.

## Writing style

Use short sentences and familiar language.

Prefer:

- “No fatigue was detected”
- “Fatigue was detected”
- “The movement was classified as correct”
- “The movement may need correction”
- “The physiotherapist agreed with the AI result”
- “The physiotherapist changed the AI result”

Avoid placing internal tool names, filenames, model filenames, database paths, or MCP implementation details in the main summary.

Technical information may appear only in a separate optional technical-details section.

## Safety and data rules

- Use only the authorised records supplied by MCP.
- Never describe a simulator or training record as a live patient recording.
- Never request or process real patient-identifying information.
- Never expose one patient's data while analysing another patient.
- Never invent missing results, measurements, reviews, or comments.
- Never provide a diagnosis.
- Never prescribe or change a rehabilitation plan.
- Never change Model 1 output automatically.
- Never call `record_review_decision` during automatic analysis.
- Call `record_review_decision` only after the physiotherapist explicitly submits a review through the review interface.
- Treat knowledge-base content as general explanatory information, not patient-specific evidence.
- Report tool failures and missing data honestly.

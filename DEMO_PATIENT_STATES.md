# Patient-view demonstration states

Use patient ID `P006`, then switch between the three available sessions:

| Session | Visual status | Patient-facing outcome |
| --- | --- | --- |
| `S001` | Green - Good to continue | Review completed; 0.0% fatigue-pattern windows and 86.7% movement consistency. |
| `S002` | Yellow - Review pending | The recording was received; percentages remain hidden until professional review. |
| `S003` | Red - Physiotherapist follow-up needed | Review completed; 73.3% fatigue-pattern windows and 53.3% movement consistency. Contact the physiotherapist before the next exercise session. |

These are counts across 15 analysed windows, not claims of clinical severity or
certainty. The stored window scores are direct outputs from the current
forearm or quadriceps CNN checkpoints. The session stream is a composite input
used to exercise the end-to-end pipeline; the dashboard does not manually set
the percentages.

For each session, ask:

> Summarise this session, explain its review status, and tell me what happens next.

The patient assistant receives only patient-appropriate reviewed information. It
does not expose raw CNN probabilities or present an unreviewed model flag as
clinical feedback.

For the care-team demonstration, open the same sessions as **Care-team
clinician**. That view adds Model 1 window percentages, fatigue onset, review
coverage, session trends, professional-review status and coordination actions.
It intentionally contains more operational evidence than the patient view.

## Publish the prepared states

Deploy the MCP data service first:

```powershell
cd <path-to-kinesis-backend>
.\deploy_public_backend.ps1
```

Then deploy the updated web application:

```powershell
cd <path-to-kinesis-public-demo>
.\deploy_public_dashboard.ps1
```

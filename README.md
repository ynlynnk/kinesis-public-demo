# KINESIS rehabilitation prototype

KINESIS is a Streamlit prototype for reviewing rehabilitation sessions from
sEMG and IMU signals. Model 1 provides movement and fatigue-pattern outputs.
Model 2 retrieves authorised session evidence and turns it into a role-based
explanation for physiotherapists, care-team clinicians, and patients.

The repository contains source code, tests, small demonstration records, model
checkpoints, and evaluation reports. Large raw training datasets, API keys,
cloud credentials, and machine-specific configuration are intentionally not
included.

## Run the dashboard locally

Install Python 3.11 or 3.12, then run these commands from PowerShell:

```powershell
git clone https://github.com/ynlynnk/kinesis-public-demo.git
cd kinesis-public-demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-ui.txt
python -m streamlit run app.py
```

Open the local address shown by Streamlit. The dashboard can be explored
without an external language-model key; in that mode it uses its local safety
summary.

## Enable Model 2 (Gemini or Claude)

Without a key everything runs; the AI panels then use local, rule-based
summaries. To let Model 2 write the progress reports and answer in the chat,
use the included launcher. It asks for a Gemini API key (from Google AI
Studio) without showing it and keeps it only for this run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_model2_gemini.ps1 -SkipInstall
```

Or set the key yourself before starting Streamlit:

```powershell
$env:GEMINI_API_KEY = "your-key"            # or $env:ANTHROPIC_API_KEY for Claude
$env:KINESIS_MODEL2_PROVIDER = "gemini"     # gemini | claude | auto
python -m streamlit run app.py
```

Optional: `KINESIS_GEMINI_MODEL` (default `gemini-3.1-flash-lite`) and
`KINESIS_CLAUDE_MODEL`. No MCP service is needed (the MCP-based "AI session
assistant" workspace was removed). See `MODEL2_GEMINI_SETUP.md`. Never commit an
API key, service-account file, `.env` file, or patient-identifying data.

## Run the tests

```powershell
$env:PYTHONPATH = "$PWD;$PWD\agent"
python -m unittest discover -q
```

## Optional live sensor input

Set `KINESIS_SENSOR_FEED_URL` to a service that follows the schema in
`FIREBASE_REALTIME_DATA_CONTRACT.md`. If it is unset, the repository's
controlled demonstration records remain available for interface testing.

## Specialist review of KINESIS app sessions

The KINESIS patient app (repository `KINESIS`, folder `host/`) is the
patient-facing side; this dashboard is the specialist side. They share one
folder, so keep the two repositories next to each other:

```
Desktop/KINESIS/
    KINESIS/                  patient app (PyQt6)
    kinesis-public-demo/      this dashboard
        report/<patient>/se<session log>/
            manifest.json  report.json  report.html          summary, criteria, fatigue, signals
            emg.csv  imu.csv  reps.csv  events.csv  ...      the raw recording
            review.json  patient_log.json  patient_log.md    written by this dashboard
            patient_read.json                                written by the app when the patient reads it
```

The path is `publish.dir` in the patient app's `host/config.yaml`
(`../../kinesis-public-demo/report`). Set `KINESIS_REPORT_DIR` to read another
location. `report/` holds patient data and is git-ignored.

### Execution steps

1. **Record a session in the patient app** (setup in the KINESIS README):

   ```powershell
   cd KINESIS\host
   python run_gui.py --sim --windowed     # virtual patient; or python run_gui.py with the sensor
   ```

   Every summary, also **Skip to results (demo)**, is sent to `report/<patient>/`
   and the app shows "Sent to your physiotherapist for review".
2. *(Optional)* From `KINESIS\host`, send sessions recorded before this existed,
   or create a simulated patient with six sessions for a demo:

   ```powershell
   python publish_sessions.py yk --dry-run   # preview, then without --dry-run to send
   python make_demo_history.py               # patient DEMO_sim, delete its folder to remove it
   ```

3. **Start the dashboard** from this folder: `python -m streamlit run app.py`
   (or the Model 2 launcher above). Open **Specialist review (KINESIS app)**
   and pick the patient and session in the sidebar; patients at risk are marked.
4. **Evaluate** the patient:
   - **AI progress report** – session k of n, rehab day, timeframe, sessions per
     week, trend per metric, a rule-based trajectory and risk tag (HIGH RISK /
     AT RISK / MONITOR / NO FLAGS) with the evidence for each flag.
     **Generate AI report** lets Model 2 reason over the whole history through
     function calls (overview, history, session detail, fatigue history, risk
     flags, knowledge) and lists every call it made.
   - **Overview**, **Signals** (whole-session IMU angle, sEMG envelope and raw
     sEMG, repetitions shaded green/red), **Repetitions** and **Fatigue**.
   - **Full report** – the app's session report with the rehabilitation record,
     progress charts and the AI summary added; downloadable as one HTML file.
   - **💬 Ask the agent** (floating button, bottom-right, on every tab) – a chat
     with memory of all the patient's sessions: it looks sessions up, compares
     them and traces faults and fatigue, citing sessions and the `knowledge/`
     library. Tables are filled from the record, not by the model, and the
     panel widens to fit them (**Shrink** restores it). It is a reference only:
     you confirm this before first use, AI replies are marked "verify before
     acting", and any number not found in the record is flagged.
5. **Review & patient log** – save the decision and clinical notes, draft the
   plain-language log, edit it and **Release to patient**.
6. **The patient reads it in the KINESIS app**: the next time they enter their
   name, a "From your physiotherapist" page opens first; older reviews stay under
   **Messages from your physiotherapist**. Once opened, this tab shows "Read by
   the patient".

Only relative labels (S1..Sn, rehab day N) reach the language model; no patient
name, folder or date is sent. Without a key the AI panels use local evidence
summaries. AI output is decision support: the diagnosis and clinical decisions
stay with the specialist.

## Main folders

- `agent/` — Model 2 retrieval, grounding, role controls, and tests.
- `artifacts/models/` — packaged Model 1 checkpoints used by the prototype.
- `evaluation/` — Model 1 and Model 2 evaluation utilities.
- `knowledge/` — approved general rehabilitation references.
- `training/` — reproducible training scripts; raw datasets are downloaded
  separately from the cited public sources.
- `outputs/evaluation/` — exported metrics and reports.

## Cloud deployment

The deployment script is account-independent. Set your own values before use:

```powershell
$env:KINESIS_GCP_PROJECT_ID = "your-google-cloud-project-id"
$env:KINESIS_MCP_URL = "https://your-mcp-service.example/mcp"
.\deploy_public_dashboard.ps1
```

The project is a demonstration and decision-support prototype. Model outputs
are not a diagnosis and patient-facing feedback remains subject to
physiotherapist review.

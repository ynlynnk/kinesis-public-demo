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

## Enable Model 2 with Gemini

Create a Gemini API key in Google AI Studio, then use the included launcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_model2_gemini.ps1 -SkipInstall
```

The launcher prompts for the key without displaying it and keeps it only in the
current process. It defaults to a local MCP service at
`http://127.0.0.1:8000/mcp`. To use another authorised MCP service:

```powershell
$env:KINESIS_MCP_URL = "https://your-mcp-service.example/mcp"
powershell -ExecutionPolicy Bypass -File .\run_model2_gemini.ps1 -SkipInstall
```

See `MODEL2_GEMINI_SETUP.md` for deployment guidance. Never commit an API key,
service-account file, `.env` file, or patient-identifying data.

This repository contains the MCP client and Model 2 grounding logic, but not
the separately deployed MCP data-service server. Without that service, the
dashboard and Model 1 still run and Model 2 uses the local safety summary. To
test the fully connected Model 2 workflow, provide an authorised MCP endpoint
through `KINESIS_MCP_URL` and use your own Gemini API key.

## Run the tests

```powershell
$env:PYTHONPATH = "$PWD;$PWD\agent"
python -m unittest discover -q
```

## Optional live sensor input

Set `KINESIS_SENSOR_FEED_URL` to a service that follows the schema in
`FIREBASE_REALTIME_DATA_CONTRACT.md`. If it is unset, the repository's
controlled demonstration records remain available for interface testing.

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

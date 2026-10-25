# KINESIS Model 2 — Gemini setup

Model 1 remains responsible for movement classifications, fatigue-pattern
scores, signal measurements, and confidence values. Model 2 retrieves only the
authorised evidence needed for the current question through MCP and asks Gemini
to turn that evidence into a role-appropriate explanation.

Gemini cannot update Model 1 values or physiotherapist review records. If the
Gemini request fails, the dashboard returns the existing local deterministic
summary instead of becoming unavailable.

## 1. Create a Gemini API key

1. Open [Google AI Studio API keys](https://aistudio.google.com/apikey).
2. Sign in with the Google account that will own the KINESIS demonstration.
3. Select **Create API key** and choose or create a Google Cloud project.
4. Copy the key and store it in a password manager.

Do not paste the key into source code, Git, screenshots, documents, or chat
messages. The Gemini free tier has quotas and is not unlimited. Use only
synthetic or de-identified session information on the free tier.

## 2. Start a private local test

Open PowerShell and run:

```powershell
cd <path-to-kinesis-public-demo>
powershell -ExecutionPolicy Bypass -File .\run_model2_gemini.ps1
```

When prompted, paste the Gemini API key and press Enter. The prompt hides the
key, sets it only for the launched dashboard process, and removes it when the
dashboard stops. It is not written into the repository. On later runs, use
`-SkipInstall` to avoid checking the dependencies again:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_model2_gemini.ps1 -SkipInstall
```

## 3. Verify Model 2 in the dashboard

1. Open **AI session assistant**.
2. Confirm the page says **AI session assistant online**.
3. Select **Physiotherapist**.
4. Select one available patient session.
5. Ask: `Summarise the fatigue and movement findings and compare them with earlier sessions.`
6. Select **Ask assistant**.
7. Confirm the result is labelled **AI-generated session answer**.

The response should explain only values already returned by Model 1 and MCP.
It must not invent a new fatigue score, movement measurement, diagnosis, or
exercise prescription.

If the page displays **Local session summary**, expand or inspect the server
log. The usual causes are an invalid key, exhausted free-tier quota, an invalid
model name, or a temporary network error. The dashboard deliberately falls
back to its local summary in these cases.

## 4. Stop the local test

Press `Ctrl+C` in the PowerShell window. The launcher removes the temporary key
and Model 2 environment settings automatically.

## 5. Configure the deployed Cloud Run dashboard

Use Google Cloud Secret Manager; do not place the key in
`deploy_public_dashboard.ps1`.

1. Open the Google Cloud Console for the project that will host KINESIS.
2. Open **Security → Secret Manager**.
3. Create a secret named `gemini-api-key` and paste the key as its first
   version.
4. Give the `kinesis-web` Cloud Run service account the role
   **Secret Manager Secret Accessor** for this secret.
5. Attach the secret and provider settings:

```powershell
gcloud run services update kinesis-web `
  --project YOUR_PROJECT_ID `
  --region asia-southeast1 `
  --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" `
  --update-env-vars "KINESIS_MODEL2_PROVIDER=gemini,KINESIS_GEMINI_MODEL=gemini-3.1-flash-lite"
```

6. Deploy the updated dashboard code:

```powershell
.\deploy_public_dashboard.ps1
```

## 6. Provider behaviour

- `KINESIS_MODEL2_PROVIDER=gemini` uses `GEMINI_API_KEY`.
- `KINESIS_MODEL2_PROVIDER=claude` retains the existing Anthropic integration.
- `KINESIS_MODEL2_PROVIDER=local` disables external language-model calls.
- `KINESIS_MODEL2_PROVIDER=auto` prefers Gemini, then Claude, then the local
  deterministic fallback.

The recommended demonstration configuration is Gemini with
`gemini-3.1-flash-lite`.

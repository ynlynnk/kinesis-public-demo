# KINESIS Model 2 — Claude setup

Gemini is also supported. For the free-tier Gemini path, use
[`MODEL2_GEMINI_SETUP.md`](MODEL2_GEMINI_SETUP.md).

Model 1 calculates the movement and fatigue results. Model 2 does not replace
the CNN. A physiotherapist asks a question, MCP retrieves the relevant
authorised evidence, and Claude turns those verified findings into a short
plain-language answer.

The dashboard works without Claude. When no key is configured, it uses a local
rule-based summary. When the key is configured, the page reports
`AI-generated session summary` as the generation mode.

## 1. Create and fund an Anthropic API account

1. Open the [Anthropic Console](https://console.anthropic.com/).
2. Create or select an organisation.
3. Open **Settings → Billing**, choose **Buy credits**, and purchase a small
   amount of prepaid API credit.
4. Open **Settings → API Keys** and create a key for KINESIS.

A paid Claude.ai chat subscription does not include API usage. Console API
credit is purchased separately. Current billing guidance is available in the
[Anthropic billing help](https://support.anthropic.com/en/articles/8977456-how-do-i-pay-for-my-api-usage),
and current token prices are listed on the
[Claude pricing page](https://platform.claude.com/docs/en/about-claude/pricing).

Do not paste the key into source code, Git, screenshots, or chat messages.

## 2. Test locally in PowerShell

From the `KINESIS_PUBLIC_DEMO` folder:

```powershell
$env:ANTHROPIC_API_KEY = "paste-your-key-in-your-own-terminal"
$env:KINESIS_CLAUDE_MODEL = "claude-sonnet-5"
$env:KINESIS_MCP_URL = "http://127.0.0.1:8000/mcp"
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

Model 2 obtains authorised session evidence through the MCP URL above. If a
local MCP service is used instead, set `KINESIS_MCP_URL` to that service's
`/mcp` endpoint before starting Streamlit.

Open **AI session assistant**, select **Physiotherapist**, choose a session,
enter a question, and select **Ask assistant**. A successful API call is
labelled `AI-generated session summary`. If the call fails, the dashboard
stays available and reports `Local session summary`.

Remove the key from the current terminal after testing:

```powershell
Remove-Item Env:ANTHROPIC_API_KEY
```

## 3. Configure the deployed Cloud Run service

Store the key in Google Secret Manager rather than in the deployment script.
For example, create a secret named `anthropic-api-key`, grant the Cloud Run
service account **Secret Manager Secret Accessor**, then attach it to the
existing service:

```powershell
gcloud run services update kinesis-web `
  --project YOUR_PROJECT_ID `
  --region asia-southeast1 `
  --set-secrets "ANTHROPIC_API_KEY=anthropic-api-key:latest" `
  --update-env-vars "KINESIS_CLAUDE_MODEL=claude-sonnet-5"
```

The current Model 2 request sends only a small, de-identified structured
summary. It does not send the raw sEMG window, patient ID, reviewer ID, or review
comment to Claude. Read-only training records remain separate from live-device
sessions.

[CmdletBinding()]
param(
    [string]$SensorFeedUrl = "",
    [string]$ProjectId = $env:KINESIS_GCP_PROJECT_ID,
    [string]$Region = $(if ($env:KINESIS_GCP_REGION) { $env:KINESIS_GCP_REGION } else { "asia-southeast1" }),
    [string]$ServiceName = $(if ($env:KINESIS_GCP_SERVICE) { $env:KINESIS_GCP_SERVICE } else { "kinesis-web" }),
    [string]$McpUrl = $env:KINESIS_MCP_URL
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectId)) {
    throw "Set KINESIS_GCP_PROJECT_ID or pass -ProjectId before deploying."
}
if ([string]::IsNullOrWhiteSpace($McpUrl)) {
    throw "Set KINESIS_MCP_URL or pass -McpUrl before deploying."
}

$image = "$Region-docker.pkg.dev/$ProjectId/cloud-run-source-deploy/$($ServiceName):latest"
$environmentVariables = "KINESIS_MCP_URL=$McpUrl,KINESIS_PUBLIC_DEMO=true"
if ($SensorFeedUrl) {
    $environmentVariables += ",KINESIS_SENSOR_FEED_URL=$SensorFeedUrl"
}

$gcloudCommand = Get-Command gcloud -ErrorAction SilentlyContinue
if ($gcloudCommand) {
    $gcloud = $gcloudCommand.Source
} else {
    $gcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
}

if (-not (Test-Path -LiteralPath $gcloud)) {
    throw "Google Cloud CLI was not found. Open the same PowerShell terminal where gcloud worked previously."
}

Push-Location $PSScriptRoot
try {
    Write-Host "Building the updated KINESIS dashboard..." -ForegroundColor Cyan
    & $gcloud builds submit `
        --config cloudbuild-ui.yaml `
        --region $Region `
        --project $ProjectId `
        .

    if ($LASTEXITCODE -ne 0) {
        throw "The Cloud Build step failed. The current public website was not changed."
    }

    Write-Host "Deploying the new revision to the existing public service..." -ForegroundColor Cyan
    & $gcloud run deploy $ServiceName `
        --project $ProjectId `
        --region $Region `
        --image $image `
        --allow-unauthenticated `
        --update-env-vars $environmentVariables `
        --port 8080 `
        --min 0 `
        --max 1 `
        --memory 1Gi `
        --cpu 1 `
        --timeout 3600 `
        --concurrency 20 `
        --session-affinity

    if ($LASTEXITCODE -ne 0) {
        throw "The Cloud Run deployment failed. Check the error shown above."
    }

    $publicUrl = & $gcloud run services describe $ServiceName `
        --project $ProjectId `
        --region $Region `
        --format "value(status.url)"

    Write-Host "Deployment complete." -ForegroundColor Green
    Write-Host "Public dashboard: $publicUrl" -ForegroundColor Green
} finally {
    Pop-Location
}

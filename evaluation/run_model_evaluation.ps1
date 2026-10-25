[CmdletBinding()]
param(
    [switch]$LiveModel2,
    [string]$BackendRoot = $env:KINESIS_BACKEND_ROOT,
    [string]$GoogleCloudProjectId = $env:KINESIS_GCP_PROJECT_ID
)

$ErrorActionPreference = "Stop"
$publicRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $publicRoot
if (-not $BackendRoot) {
    $BackendRoot = Join-Path $workspaceRoot "KINESIS_AI_DEMO"
}
$outputDirectory = Join-Path $publicRoot "outputs\evaluation"

$python = $null
$backendPython = Join-Path $BackendRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $backendPython) {
    # Prefer the KINESIS environment verified with torch, sklearn and the web app.
    $python = $backendPython
}
if (-not $python -and $env:VIRTUAL_ENV) {
    $candidate = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate) {
        $python = $candidate
    }
}
if (-not $python) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $python = $pythonCommand.Source
    }
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python was not found. Open the VS Code terminal where the KINESIS environment works."
}

$oldPythonPath = $env:PYTHONPATH
$oldProvider = $env:KINESIS_MODEL2_PROVIDER
$oldModel = $env:KINESIS_GEMINI_MODEL
$temporarySecretLoaded = $false
$pythonPaths = @(
    $publicRoot,
    (Join-Path $publicRoot "agent")
)
$env:PYTHONPATH = ($pythonPaths -join ";")

try {
    Write-Host "Exporting Model 1 held-out CNN metrics..." -ForegroundColor Cyan
    & $python (Join-Path $PSScriptRoot "model1_accuracy.py") `
        --output-directory $outputDirectory *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Model 1 evaluation failed."
    }

    Write-Host "Evaluating repeated-window forearm session decisions..." -ForegroundColor Cyan
    & $python (Join-Path $PSScriptRoot "forearm_session_accuracy.py") *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Forearm session-decision evaluation failed."
    }

    $improvementExporter = Join-Path $PSScriptRoot "export_model1_improvements.py"
    $personalizedMetrics = Join-Path $outputDirectory "quadriceps_movement_personalized_metrics.json"
    $fatigueCandidate = Join-Path $publicRoot "artifacts\models\candidates\kinesis_quadriceps_fatigue_feature_cnn_v2.pt"
    $forearmAugmentedCandidate = Join-Path $publicRoot "artifacts\models\candidates\kinesis_forearm_movement_augmented_cnn_v2.pt"
    $forearm40Candidate = Join-Path $publicRoot "artifacts\models\candidates\kinesis_forearm_movement_ckwc76xr2z_cnn_v1.pt"
    if (
        (Test-Path -LiteralPath $personalizedMetrics) -and
        (Test-Path -LiteralPath $fatigueCandidate) -and
        (Test-Path -LiteralPath $forearmAugmentedCandidate) -and
        (Test-Path -LiteralPath $forearm40Candidate)
    ) {
        & $python $improvementExporter *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "Model 1 candidate comparison failed."
        }
    }

    $model2Arguments = @(
        (Join-Path $PSScriptRoot "model2_evaluation.py"),
        "--backend-root", $BackendRoot,
        "--output-directory", $outputDirectory
    )
    if ($LiveModel2) {
        if (-not $env:GEMINI_API_KEY) {
            if ([string]::IsNullOrWhiteSpace($GoogleCloudProjectId)) {
                throw "Set KINESIS_GCP_PROJECT_ID or pass -GoogleCloudProjectId to load the Gemini secret."
            }
            $gcloudCommand = Get-Command gcloud -ErrorAction SilentlyContinue
            if ($gcloudCommand) {
                $gcloud = $gcloudCommand.Source
            } else {
                $gcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
            }
            if (-not (Test-Path -LiteralPath $gcloud)) {
                throw "Google Cloud CLI was not found, so the Gemini secret could not be loaded."
            }
            $secretValue = & $gcloud secrets versions access latest `
                --secret gemini-api-key `
                --project $GoogleCloudProjectId
            if ($LASTEXITCODE -ne 0 -or -not $secretValue) {
                throw "The Gemini key could not be read from Secret Manager."
            }
            $env:GEMINI_API_KEY = ($secretValue -join "").Trim()
            $temporarySecretLoaded = $true
        }
        $env:KINESIS_MODEL2_PROVIDER = "gemini"
        if (-not $env:KINESIS_GEMINI_MODEL) {
            $env:KINESIS_GEMINI_MODEL = "gemini-3.1-flash-lite"
        }
        $model2Arguments += "--live"
        Write-Host "Running three live grounded-response checks..." -ForegroundColor Cyan
    } else {
        Write-Host "Running offline Model 2 planner and guardrail checks..." -ForegroundColor Cyan
    }

    & $python @model2Arguments *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Model 2 evaluation failed."
    }

    Write-Host "Evaluation complete." -ForegroundColor Green
    Write-Host "Model 1 report: $outputDirectory\model1_cnn_metrics.md"
    Write-Host "Model 2 report: $outputDirectory\model2_evaluation.md"
    Write-Host "Raw JSON files are in the same folder."
} finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:KINESIS_MODEL2_PROVIDER = $oldProvider
    $env:KINESIS_GEMINI_MODEL = $oldModel
    if ($temporarySecretLoaded) {
        Remove-Item Env:GEMINI_API_KEY -ErrorAction SilentlyContinue
    }
}

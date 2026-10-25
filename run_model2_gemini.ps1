[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [string]$McpUrl = $(if ($env:KINESIS_MCP_URL) { $env:KINESIS_MCP_URL } else { "http://127.0.0.1:8000/mcp" })
)

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$oldMcpUrl = $env:KINESIS_MCP_URL

Push-Location $projectDirectory

try {
    $secureKey = Read-Host "Paste the Gemini API key (the text will stay hidden)" -AsSecureString
    $credential = [System.Management.Automation.PSCredential]::new(
        "Gemini",
        $secureKey
    )
    $plainKey = $credential.GetNetworkCredential().Password

    if ([string]::IsNullOrWhiteSpace($plainKey)) {
        throw "No Gemini API key was entered."
    }

    $env:GEMINI_API_KEY = $plainKey
    $env:KINESIS_MODEL2_PROVIDER = "gemini"
    $env:KINESIS_GEMINI_MODEL = "gemini-3.1-flash-lite"
    $env:KINESIS_MCP_URL = $McpUrl

    if (-not $SkipInstall) {
        Write-Host "Checking KINESIS dashboard dependencies..." -ForegroundColor Cyan
        & python -m pip install -r requirements-ui.txt
        if ($LASTEXITCODE -ne 0) {
            throw "The Python dependencies could not be installed."
        }
    }

    Write-Host "Starting KINESIS with Gemini Model 2..." -ForegroundColor Green
    Write-Host "Keep this PowerShell window open. Press Ctrl+C to stop." -ForegroundColor DarkGray
    & python -m streamlit run app.py
    if ($LASTEXITCODE -ne 0) {
        throw "The KINESIS dashboard stopped with an error."
    }
}
finally {
    Remove-Item Env:GEMINI_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:KINESIS_MODEL2_PROVIDER -ErrorAction SilentlyContinue
    Remove-Item Env:KINESIS_GEMINI_MODEL -ErrorAction SilentlyContinue
    if ($null -eq $oldMcpUrl) {
        Remove-Item Env:KINESIS_MCP_URL -ErrorAction SilentlyContinue
    } else {
        $env:KINESIS_MCP_URL = $oldMcpUrl
    }
    $plainKey = $null
    $credential = $null
    $secureKey = $null
    Pop-Location
}

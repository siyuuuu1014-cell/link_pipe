param([string]$CondaEnvironment = "yolo")
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

$Backend = Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "start_backend.ps1"), "-CondaEnvironment", $CondaEnvironment -WindowStyle Hidden -PassThru
try {
    $Healthy = $false
    for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
        try {
            $Response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2
            if ($Response.status -eq "ok") { $Healthy = $true; break }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $Healthy) { throw "Backend health check timed out." }
    $Frontend = Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "start_frontend.ps1") -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 3
    Start-Process "http://127.0.0.1:5173"
    Write-Host "System started. Backend PID=$($Backend.Id), Frontend PID=$($Frontend.Id)"
} catch {
    if (-not $Backend.HasExited) { Stop-Process -Id $Backend.Id }
    throw
}

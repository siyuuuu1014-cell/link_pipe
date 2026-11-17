param(
    [string]$CondaEnvironment = "yolo",
    [switch]$UseVenv
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($UseVenv) {
    $VenvPath = Join-Path $ProjectRoot ".venv"
    if (-not (Test-Path -LiteralPath $VenvPath)) {
        python -m venv $VenvPath
    }
    $Python = Join-Path $VenvPath "Scripts\python.exe"
} else {
    conda run -n $CondaEnvironment python --version
    $Python = "conda"
}

if ($UseVenv) {
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -e ".[dev]"
} else {
    conda run -n $CondaEnvironment python -m pip install --upgrade pip
    conda run -n $CondaEnvironment python -m pip install -e ".[dev]"
}

npm.cmd --prefix frontend install
if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
}

$Directories = @(
    "data\processed", "data\external_test", "data\demo",
    "models\base", "models\trained", "models\deployed", "models\metrics",
    "storage\uploads", "storage\results", "storage\videos", "storage\reports", "storage\database"
)
foreach ($Directory in $Directories) {
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
}

if ($UseVenv) {
    & $Python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
} else {
    conda run -n $CondaEnvironment python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
}
Write-Host "Environment setup completed."

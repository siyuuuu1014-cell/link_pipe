param(
    [string]$CondaEnvironment = "yolo",
    [int]$Epochs = 150,
    [int]$Batch = 8,
    [switch]$Smoke
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
function Invoke-Stage([string]$Name, [string[]]$Arguments) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & conda run -n $CondaEnvironment python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed. Remaining stages were stopped." }
}

Invoke-Stage "Inspect raw data" @("-m", "training.inspect_dataset")
Invoke-Stage "Preprocess data" @("-m", "training.preprocess_dataset", "--force")
Invoke-Stage "Audit processed data" @("-m", "training.audit_dataset")
$TrainFullArgs = @("-m", "training.train_full", "--epochs", "$Epochs", "--batch", "$Batch")
$TrainLoraArgs = @("-m", "training.train_lora", "--epochs", "$Epochs", "--batch", "$Batch")
if ($Smoke) { $TrainFullArgs += "--smoke"; $TrainLoraArgs += "--smoke" }
Invoke-Stage "Full fine-tuning" $TrainFullArgs
Invoke-Stage "LoRA fine-tuning" $TrainLoraArgs

$Base = if ($Smoke) { "models/trained/smoke" } else { "models/trained" }
$Suffix = if ($Smoke) { "_smoke" } else { "" }
$EvalFullArgs = @("-m", "training.evaluate_internal", "--model", "$Base/yolo11s_full$Suffix/weights/best.pt", "--kind", "full")
$EvalLoraArgs = @("-m", "training.evaluate_internal", "--model", "$Base/yolo11s_lora$Suffix/weights/best.pt", "--kind", "lora")
$CompareArgs = @("-m", "training.compare_models")
if ($Smoke) { $EvalFullArgs += "--smoke"; $EvalLoraArgs += "--smoke"; $CompareArgs += "--smoke" }
Invoke-Stage "Evaluate full fine-tuning" $EvalFullArgs
Invoke-Stage "Evaluate LoRA fine-tuning" $EvalLoraArgs
Invoke-Stage "Compare models" $CompareArgs

if (-not $Smoke) {
    Invoke-Stage "Export deployment model" @("-m", "training.export_model")
} else {
    Write-Warning "Smoke weights are never written to models/deployed/best.pt."
}

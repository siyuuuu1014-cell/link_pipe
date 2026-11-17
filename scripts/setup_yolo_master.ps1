param([string]$CondaEnvironment = "yolo")
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VendorPath = Join-Path $ProjectRoot "vendor\YOLO-Master"
Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $VendorPath)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $VendorPath) -Force | Out-Null
    git clone --depth 1 https://github.com/Tencent/YOLO-Master.git $VendorPath
}
conda run -n $CondaEnvironment python -m pip install -e $VendorPath --no-deps --no-build-isolation
conda run -n $CondaEnvironment python -c "from ultralytics.cfg import DEFAULT_CFG_DICT; keys=('lora_r','lora_alpha','lora_dropout','lora_gradient_checkpointing'); assert all(k in DEFAULT_CFG_DICT for k in keys); print('YOLO-Master LoRA ready')"

# 管道泄漏多目标智能检测系统

基于 YOLO11 的 POC/客户演示系统，用于在可见光图片和 MP4 视频中定位管道裂纹、可见水体和疑似泄漏，并提供风险分级、固定安全建议、历史记录和可打印报告。

## 能力边界

本项目仅处理外部可见视觉特征，不能检测不可见天然气泄漏、内部堵塞或内部腐蚀，不能测量泄漏量、确认泄漏介质或替代工业安全验收。页面和报告不会使用“确认无泄漏”“绝对安全”等表述。

> 本结果仅基于可见光图像中的外部视觉特征生成，不能直接确认泄漏介质、内部腐蚀、内部堵塞或精确故障原因，不能替代压力、流量、可燃气体、无损检测及现场人工安全检查。

## 目录结构

```text
data/                 原始、处理后、外部测试和演示数据
models/               基础权重、训练结果、部署模型和真实指标
training/             检查、预处理、训练、评估、比较和导出脚本
notebooks/            相互隔离的 LoRA 冒烟与正式训练 Notebook
backend/              FastAPI、SQLite、推理、风险、建议、视频和报告
frontend/             Vue 3 + TypeScript 客户演示页面
storage/              上传、结果、视频、报告和 SQLite 数据库
scripts/              Windows 环境、训练、构建和启动脚本
```

## 环境要求

- Windows 11
- 当前使用 Conda 环境 `yolo`，Python 3.13
- NVIDIA RTX 5060 Laptop 8GB，CUDA 可用
- Node.js 24 和 npm
- Windows 训练必须使用 `workers=0`

```powershell
conda activate yolo
python -m pip install -e ".[dev]"
npm.cmd --prefix frontend install
```

安装支持 LoRA 的腾讯 YOLO-Master fork：

```powershell
.\scripts\setup_yolo_master.ps1
```

不要在安装 YOLO-Master 后执行 `pip install -U ultralytics`，否则会换回不包含本项目 LoRA 参数的标准包。

检查 CUDA：

```powershell
conda run -n yolo python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 数据集与预处理

原始 Roboflow 数据放在：

```text
data/pipeline_leakage1/
```

原始类别顺序由 `data.yaml` 动态读取。预处理会将多边形转换为轴对齐 BBox，删除所有 `no leak` 框但保留图片和空标签，并使用 seed=42 将 502 张原验证图片多标签分层拆成 251 张 valid 和 251 张 test。

```powershell
conda run -n yolo python -m training.inspect_dataset
conda run -n yolo python -m training.preprocess_dataset --force
conda run -n yolo python -m training.audit_dataset
```

处理后固定类别为：

```yaml
0: crack
1: water
2: leak
```

报告位于 `models/metrics/dataset_inspection.json`、`preprocess_report.json`、`suspicious_samples.csv` 和 `dataset_audit.json`。审计失败时训练入口会返回非 0 状态并停止。

## 训练

训练参数位于 `training/configs/train.yaml`。默认使用 YOLO11s、640 像素、AdamW、150 epochs；8GB 显存建议 batch=8。

完整微调：

```powershell
conda run -n yolo python -m training.train_full --epochs 150 --batch 8
```

LoRA 微调：

```powershell
conda run -n yolo python -m training.train_lora --epochs 150 --batch 8
```

3 epochs 冒烟验证：

```powershell
.\scripts\train_all.ps1 -Smoke -Epochs 3 -Batch 8
```

冒烟权重只写入 `models/trained/smoke/`，不能作为最终客户模型，也不会生成正式 `selected_model.json`。

### LoRA Notebook

以下 Notebook 都选择现有 Conda 环境 `yolo` 作为 Kernel，并按顺序运行：

- `notebooks/lora_finetuning_smoke.ipynb`：3 epochs 冒烟验证，结果只写入 smoke 目录。
- `notebooks/lora_finetuning_formal.ipynb`：独立的 150 epochs 正式 LoRA 稳定训练，结果写入 `models/trained/notebook/yolo11s_lora_formal_v2/`，指标写入 `models/metrics/notebook_lora_v2_internal_metrics.json`。旧的 `yolo11s_lora_formal/` 是已暂停的问题实验，禁止续训或部署。
- `notebooks/yolo11s_lora_finetuning_v3.ipynb`：独立的 LoRA v3 优化实验，最多 100 epochs，结果写入 `models/trained/notebook/yolo11s_lora_formal_v3/`。它会在固定 test 集评估并自动生成 v3 与 v2 的指标差值，不会覆盖 v2。

LoRA v3 也可以直接从终端运行。默认命令只执行 3 epochs smoke；加入 `--formal` 才会启动正式训练：

```powershell
conda run -n yolo python -m training.train_lora_v3
conda run -n yolo python -m training.train_lora_v3 --formal --export-onnx
```

正式 Notebook 会保存 `best.pt`、LoRA adapter、测试集指标和训练清单，并支持从自身的 `last.pt` 续训。它不会覆盖主线 `models/trained/yolo11s_lora/`、`models/deployed/best.pt` 或主线模型选择结果；如需将该模型用于客户演示，仍需显式纳入主线正式比较与导出流程。

### YOLO26s 完整微调 Notebook

`notebooks/yolo26s_full_finetuning_formal.ipynb` 使用官方 YOLO26s、768 分辨率和完整微调，输出到 `models/trained/yolo26/formal/yolo26s_full_768_formal/`。它必须选择独立的 `Python (YOLO26 Project)` Kernel（解释器为 `.venv-yolo26/Scripts/python.exe`），不能选择 YOLO-Master 的 `yolo` Kernel。

如需重建这个独立环境：

```powershell
conda run -n yolo python -m venv --system-site-packages .venv-yolo26
.\.venv-yolo26\Scripts\python.exe -m pip install -r requirements-yolo26.txt
.\.venv-yolo26\Scripts\python.exe -m ipykernel install --prefix .venv-yolo26 --name yolo26-project --display-name "Python (YOLO26 Project)"
```

官方预训练权重固定放在项目根目录 `yolo26s.pt` 和 `yolo26m.pt`。不要在 Conda 环境 `yolo` 中安装或升级官方 `ultralytics`，否则会覆盖 YOLO-Master LoRA 运行时。

YOLO26 的 3 epochs、640/batch8 兼容性预检已经通过；预检结果只在 `models/trained/yolo26/smoke/`，不得部署。正式 Notebook 默认 `batch=4`，如 768 分辨率出现显存不足可降为 2。

### YOLO26s 独立 LoRA 微调

YOLO26 LoRA 使用官方 `.venv-yolo26`，不能使用 YOLO-Master 的 `yolo` 环境。它以已经完成的
`models/trained/yolo26/formal/yolo26s_full_768_formal/weights/best.pt` 为只读基座，仅训练后段
neck/特征层的卷积 LoRA Adapter。原始完整微调权重会在训练前后进行 SHA-256 校验。

先运行 3 epochs 冒烟验证：

```powershell
.\.venv-yolo26\Scripts\python.exe -m training.train_yolo26_lora --smoke --epochs 3 --imgsz 640 --batch 4 --run-name yolo26s_lora_smoke_manual
```

冒烟全部通过后再运行正式训练：

```powershell
.\.venv-yolo26\Scripts\python.exe -m training.train_yolo26_lora --epochs 150 --imgsz 768 --batch 4
```

也可以分别打开：

- `notebooks/yolo26s_lora_smoke.ipynb`
- `notebooks/yolo26s_lora_finetuning_formal.ipynb`

冒烟输出位于 `models/trained/yolo26_lora/smoke/`，正式输出位于
`models/trained/yolo26_lora/formal/yolo26s_lora_formal/`。每次训练会生成：

- `weights/best.pt`：已合并 LoRA 的标准 YOLO26 权重，可直接供后端加载；
- `weights/lora_best.pt`：保留 LoRA 分支的研究/恢复权重；
- `adapters/adapter.pt`：独立 Adapter；
- `weights/best.onnx`、固定 test 指标与 `lora_manifest.json`。

已存在同名目录时程序会拒绝覆盖。需要从未完成的 `last.pt` 继续时使用 `--resume`。冒烟权重不得部署；正式模型也只有在固定 test 集优于原始 YOLO26s 后才能手动切换部署配置。

### YOLO26s 全参数 vs LoRA 公平对照实验

`notebooks/yolo26s_full_vs_lora_experiment.ipynb` 用于比较训练方法。它与上面的“在全参模型上继续做 LoRA”不同：LoRA 对照分支和现有全参数分支都从同一个官方 `yolo26s.pt` 开始。由于官方权重为 COCO 80 类，对照 LoRA 会训练 Adapter 加新建的 3 类检测头，其余参数冻结。

已验证的 3 epochs 冒烟命令：

```powershell
.\.venv-yolo26\Scripts\python.exe -m training.train_yolo26_lora_comparison --smoke --epochs 3 --imgsz 640 --batch 4 --run-name yolo26s_lora_from_pretrained_smoke_manual
```

正式对照训练：

```powershell
.\.venv-yolo26\Scripts\python.exe -m training.train_yolo26_lora_comparison --epochs 150 --imgsz 768 --batch 4
```

正式 LoRA 完成后生成统一报告：

```powershell
.\.venv-yolo26\Scripts\python.exe -m training.compare_yolo26_full_vs_lora
```

实验产物位于 `models/experiments/yolo26s_full_vs_lora/`。比较报告包含同一固定 test 集的 Precision、Recall、mAP、每类指标，以及训练轮次、训练时间、可训练参数、Adapter 大小、合并模型大小和推理时间。

### YOLO26m 精度/速度候选 Notebook

`notebooks/yolo26m_full_finetuning_formal.ipynb` 使用相同固定数据划分完整微调官方 YOLO26m，输出到 `models/trained/yolo26m/formal/yolo26m_full_768_formal/`。默认使用 `imgsz=768`、`batch=1`、MuSGD 和最多 150 epochs；训练完成后会在固定 test 集评估，并读取 `yolo26s_full_768_internal_metrics.json` 自动比较精度与推理速度。结果不会自动部署。

当前主线已因训练耗时暂停 YOLO26m，保留脚本和第 47 epoch 断点仅供以后恢复，不再作为本轮交付依赖。

命令行入口同样与 YOLO26s 隔离：

```powershell
.\.venv-yolo26\Scripts\python.exe -m training.train_yolo26m_full --smoke --epochs 3 --imgsz 640 --batch 2
.\.venv-yolo26\Scripts\python.exe -m training.train_yolo26m_full --epochs 150 --imgsz 768 --batch 1
```

## 评估与导出

```powershell
conda run -n yolo python -m training.evaluate_internal --model models/trained/yolo11s_full/weights/best.pt --kind full
conda run -n yolo python -m training.evaluate_internal --model models/trained/yolo11s_lora/weights/best.pt --kind lora
conda run -n yolo python -m training.compare_models
conda run -n yolo python -m training.export_model
```

模型选择优先 Recall；差异小于 0.02 时依次比较 mAP50、mAP50-95、推理时间和文件大小。只有两组正式指标都存在时才允许导出 `models/deployed/best.pt`。ONNX 或 TensorRT 导出失败不会影响 `.pt` 推理。

外部测试：

```powershell
conda run -n yolo python -m training.evaluate_external --model models/deployed/best.pt
```

无标签外部数据只生成“外部图片定性测试”，不计算 Precision、Recall 或 mAP。

## 启动系统

所有命令均在项目根目录 `D:\dev\projects\yolo_train` 下执行。

### 1. 首次启动准备

确认 Conda 环境、Python 依赖和前端依赖已经安装：

```powershell
conda activate yolo
python -m pip install -e ".[dev]"
npm.cmd --prefix frontend install
```

首次启动时创建本地环境配置：

```powershell
Copy-Item .env.example .env
```

默认加载的模型是 `models/deployed/best.pt`。当前该文件已经部署为 **YOLO11s LoRA v3 正式模型**，配套 ONNX 文件为 `models/deployed/best.onnx`；部署来源、指标和文件校验值记录在 `models/deployed/model_info.json`。`.env` 保持以下配置即可：

```dotenv
YOLO_MODEL_PATH=models/deployed/best.pt
YOLO_DEVICE=0
```

后端必须在 `yolo` Conda 环境中启动，因为 YOLO11s LoRA v3 依赖项目内的 YOLO-Master LoRA 模型定义。切换部署模型或修改 `.env` 后，需要重启后端进程才能加载新权重。

未生成模型时系统仍可启动并浏览页面，但图片和视频检测不可用，健康检查接口会通过 `model_available=false` 明确说明原因。不要把 `models/deployed/smoke/best.pt` 用于正式客户演示。

### 2. 分别启动前后端

打开第一个 PowerShell 窗口启动后端：

```powershell
.\scripts\start_backend.ps1
```

打开第二个 PowerShell 窗口启动前端：

```powershell
.\scripts\start_frontend.ps1
```

两个窗口需要保持运行。修改后端代码时 Uvicorn 会自动重载，修改前端代码时 Vite 会自动刷新。

### 3. 一键启动

也可以执行以下命令自动启动后端和前端、等待后端健康检查并打开浏览器：

```powershell
.\scripts\start_all.ps1
```

如果使用的 Conda 环境名不是 `yolo`，可以显式传入：

```powershell
.\scripts\start_all.ps1 -CondaEnvironment "你的环境名"
```

### 4. 访问地址

- 前端：http://127.0.0.1:5173
- 后端：http://127.0.0.1:8000
- API 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/api/v1/health

浏览器没有自动打开时，手动访问前端地址即可。可通过健康检查中的 `model_available` 判断模型是否成功加载。

### 5. 生产构建检查

发布或演示前可执行前端生产构建：

```powershell
.\scripts\build_frontend.ps1
```

构建文件会生成在 `frontend/dist/`。该命令只负责构建前端，不会启动 FastAPI 后端。

## Ollama 可选配置

复制 `.env.example` 为 `.env`，设置：

```dotenv
ENABLE_LLM=true
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
```

Ollama 只润色固定建议，不能改变检测类别、原因或风险等级；服务不可用时自动回退固定规则。

## 测试

```powershell
conda run -n yolo python -m pytest -q
npm.cmd --prefix frontend run build
```

## 演示步骤

1. 打开系统总览，说明这是 POC 演示版本；
2. 展示正式内部测试指标；若未完成则页面明确显示“尚未完成模型评估”；
3. 上传正常管道图片，展示“当前图像中未检测到明显异常”；
4. 上传泄漏图片，展示 BBox、置信度、风险和建议；
5. 上传裂纹与水体图片，展示多类别、多目标结果；
6. 打开检测报告和历史记录；
7. 说明正式项目仍需现场 RGB/OGI、压力、流量、声学和专家复核数据。

## 常见错误

- `模型文件不存在`：尚未完成两组正式训练和模型导出；页面会保持可启动但禁用真实检测。
- CUDA OOM：把 batch 从 8 降为 4，保持 `workers=0`。
- `LoRA options missing`：运行 `scripts/setup_yolo_master.ps1`，不要升级回标准 Ultralytics。
- PowerShell 无法运行 `npm.ps1`：使用 `npm.cmd`。
- ONNX/TensorRT 导出失败：先继续使用 `models/deployed/best.pt`，导出错误记录在 `model_info.json`。
- 前端无指标：只有 150 epochs 正式评估结果才会显示；冒烟指标被隔离。

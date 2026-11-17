# Training Notebooks

## YOLO11 LoRA v3

- `yolo11s_lora_finetuning_v3.ipynb`: 独立的 YOLO11s LoRA v3 实验。正式训练最多 100 epochs，使用 640 输入、seed=42、r=16、较低学习率和提前关闭 mosaic；输出到 `models/trained/notebook/yolo11s_lora_formal_v3/`，不会覆盖 v2。
- 建议先把 Notebook 中的 `RUN_FORMAL` 改为 `False`，完成 3 epochs smoke；通过后再改回 `True` 跑正式训练。
- 正式结果会写入 `models/metrics/notebook_lora_v3_formal_internal_metrics.json`，并自动生成 `notebook_lora_v3_vs_v2.json`。

- `lora_finetuning_smoke.ipynb`：3 epochs 技术链路验证，输出到 `models/trained/smoke/`。
- `lora_finetuning_formal.ipynb`：独立的 150 epochs 正式 LoRA 稳定训练线，输出到 `models/trained/notebook/yolo11s_lora_formal_v2/`。当前配方已经通过 3 epochs 预检；旧的 `yolo11s_lora_formal/` 是已暂停的问题实验，请勿续训或部署。

打开 Notebook 后选择现有 Conda 环境 `yolo` 作为 Kernel，再按顺序运行单元格。正式 Notebook 使用保守 LoRA 配方（r=8、alpha=8、冻结 BN、关闭 AMP/RS-LoRA/alpha warmup），其权重、adapter、指标和 ONNX 都与主线隔离；若要纳入最终客户模型比较，仍需通过主线的正式评估和模型选择流程。

- `yolo26s_full_finetuning_formal.ipynb`：官方 YOLO26s、768 分辨率、完整微调的独立正式实验。必须选择 `Python (YOLO26 Project)` Kernel，输出到 `models/trained/yolo26/formal/yolo26s_full_768_formal/`。
- `yolo26s_lora_smoke.ipynb`：以已完成 YOLO26s 为只读基座的 3 epochs LoRA 链路验证，输出到 `models/trained/yolo26_lora/smoke/`，不得部署。
- `yolo26s_lora_finetuning_formal.ipynb`：YOLO26s 专用 150 epochs 正式 LoRA 微调，独立保存 Adapter、LoRA 原始权重、合并权重、ONNX 和固定 test 指标；不会覆盖完整微调基座。
- `yolo26s_full_vs_lora_experiment.ipynb`：全参数与 LoRA 的公平对照实验；两组从相同官方 `yolo26s.pt` 出发，LoRA 训练 Adapter 和重新初始化的 3 类检测头，正式完成后自动生成统一比较报告。
- `yolo26m_full_finetuning_formal.ipynb`：官方 YOLO26m、768 分辨率、完整微调的精度/速度候选实验。使用同一个 `Python (YOLO26 Project)` Kernel，默认 `batch=1`，输出到 `models/trained/yolo26m/formal/yolo26m_full_768_formal/`，并自动与 YOLO26s 固定 test 基线比较。

YOLO26 使用项目内 `.venv-yolo26` 和官方 Ultralytics 8.4.55，不能选择 `yolo` Kernel；这样可以避免覆盖 YOLO-Master 的 LoRA 环境。它的 3 epochs、640/batch8 预检已经通过，预检权重只位于 `models/trained/yolo26/smoke/`，不得部署。

YOLO26 LoRA 同样必须使用 `Python (YOLO26 Project)`。现有 YOLO11/YOLO-Master LoRA Notebook 不能加载 YOLO26 端到端检测头，不得交叉续训。

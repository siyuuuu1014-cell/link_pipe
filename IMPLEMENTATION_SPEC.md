# CODEX 实施任务书：管道泄漏多目标检测与客户演示系统

> 本文件用于直接交给 Codex 执行。  
> Codex 应按照本文件完成项目搭建、数据预处理、模型训练、评估、推理服务、前端页面、报告生成和本地部署。  
> 本项目当前使用单一公开数据集 `pipeline_leakage1`，训练多类别、多目标 YOLO 检测模型。

---

## 1. 项目目标

构建一套可直接给客户演示的“管道泄漏智能检测系统”。

客户可以通过前端页面：

1. 上传管道图片；
2. 查看原始图片和检测后图片；
3. 查看多个检测框、类别、置信度和目标数量；
4. 查看综合状态：
   - 疑似泄漏；
   - 发现裂纹风险；
   - 发现可见水体；
   - 当前图像未检测到明显异常；
5. 查看针对不同异常位置和类别的处理建议；
6. 查看模型在测试集上的 Precision、Recall、mAP50、mAP50-95；
7. 下载检测结果图片和检测报告；
8. 可选上传视频进行逐帧检测；
9. 可选调用本地大语言模型润色建议。

本项目为 POC/客户演示版本，不得声称已经达到油田正式工业验收标准。

---

## 2. 当前数据集

数据集名称：

```text
pipeline_leakage1
```

预期目录：

```text
data/pipeline_leakage1/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

数据集原始类别可能包括：

```text
crack
water
leak
no leak
```

Codex 必须先读取真实 `data.yaml`，不能假设类别 ID 顺序。

---

## 3. 最终检测类别

YOLO 目标检测模型只保留以下三个类别：

```yaml
0: crack
1: water
2: leak
```

中文显示名：

```python
CLASS_NAME_ZH = {
    "crack": "管道裂纹",
    "water": "可见水体/积水",
    "leak": "疑似泄漏",
}
```

### 3.1 `no leak` 的处理方式

`no leak` 不作为 YOLO 检测类别。

原因：

- `no leak` 是整张图片的状态，不是具有明确边界的目标；
- 若把 `no leak` 作为 BBox 类别，模型可能同时输出 `leak` 和 `no leak`；
- 正常图片应作为负样本存在。

处理规则：

```text
原始图片保留；
删除该图片中所有 no leak 对应的 BBox；
若图片删除 no leak 框后没有其他目标，则生成空标签文件。
```

前端根据 YOLO 是否输出目标，生成 `no leak` 状态：

```text
未检测到 crack、water、leak
→ 当前图像中未检测到超过阈值的明显异常
```

不得显示：

```text
确认无泄漏
管道完全正常
```

---

## 4. 技术栈

### 4.1 模型与训练

```text
Python 3.11
PyTorch
Ultralytics YOLO11
YOLO-Master（用于 LoRA）
OpenCV
Pillow
Pandas
PyYAML
```

基础模型：

```text
yolo11s.pt
```

训练两组模型：

```text
A. YOLO11s 完整微调
B. YOLO11s + LoRA 微调
```

最终使用测试集表现更好的模型进行部署。

### 4.2 后端

```text
FastAPI
Uvicorn
SQLAlchemy
SQLite
Pydantic
python-multipart
Jinja2（仅用于报告模板）
```

### 4.3 前端

```text
Vue 3
Vite
TypeScript
Element Plus
Axios
ECharts
Pinia
Vue Router
```

### 4.4 可选本地大语言模型

```text
Ollama
Qwen3:4b
```

大语言模型只负责润色结构化建议，不能直接决定检测类别、故障原因或风险等级。

---

## 5. 项目目录结构

Codex 必须创建以下项目结构：

```text
pipeline-leak-detection/
├── README.md
├── CODEX_IMPLEMENTATION_SPEC.md
├── .env.example
├── .gitignore
├── requirements.txt
├── docker-compose.yml
│
├── data/
│   ├── pipeline_leakage1/
│   ├── processed/
│   ├── external_test/
│   └── demo/
│
├── models/
│   ├── base/
│   ├── trained/
│   ├── deployed/
│   └── metrics/
│
├── training/
│   ├── inspect_dataset.py
│   ├── preprocess_dataset.py
│   ├── audit_dataset.py
│   ├── train_full.py
│   ├── train_lora.py
│   ├── evaluate_internal.py
│   ├── evaluate_external.py
│   ├── compare_models.py
│   ├── export_model.py
│   └── configs/
│       └── train.yaml
│
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── dependencies.py
│   ├── schemas/
│   ├── models/
│   ├── api/
│   │   ├── health.py
│   │   ├── detect.py
│   │   ├── videos.py
│   │   ├── metrics.py
│   │   ├── history.py
│   │   └── reports.py
│   ├── services/
│   │   ├── yolo_service.py
│   │   ├── image_service.py
│   │   ├── video_service.py
│   │   ├── risk_service.py
│   │   ├── advice_service.py
│   │   ├── llm_service.py
│   │   └── report_service.py
│   ├── templates/
│   │   └── detection_report.html
│   ├── static/
│   └── tests/
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── src/
│   │   ├── main.ts
│   │   ├── App.vue
│   │   ├── router/
│   │   ├── stores/
│   │   ├── api/
│   │   ├── types/
│   │   ├── components/
│   │   └── views/
│   │       ├── DashboardView.vue
│   │       ├── ImageDetectionView.vue
│   │       ├── VideoDetectionView.vue
│   │       ├── MetricsView.vue
│   │       ├── HistoryView.vue
│   │       └── ReportView.vue
│   └── public/
│
├── storage/
│   ├── uploads/
│   ├── results/
│   ├── videos/
│   ├── reports/
│   └── database/
│
└── scripts/
    ├── setup_windows.ps1
    ├── start_backend.ps1
    ├── start_frontend.ps1
    ├── start_all.ps1
    ├── train_all.ps1
    └── build_frontend.ps1
```

---

## 6. 第一阶段：数据检查

创建：

```text
training/inspect_dataset.py
```

功能：

1. 读取 `data/pipeline_leakage1/data.yaml`；
2. 打印原始类别名称和 ID；
3. 统计 train/valid/test 图片数量、每个类别 BBox 数量、空标签图片数量、缺失标签文件数量；
4. 检查图片与标签是否一一对应；
5. 检查标签是否为标准 YOLO 检测格式；
6. 检查坐标是否处于 `[0, 1]`；
7. 输出 JSON 报告：

```text
models/metrics/dataset_inspection.json
```

如果 `data.yaml` 不存在、类别无法解析或目录异常，应给出明确错误信息。

---

## 7. 第二阶段：自动预处理数据集

创建：

```text
training/preprocess_dataset.py
```

输入：

```text
data/pipeline_leakage1/
```

输出：

```text
data/processed/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 7.1 类别映射规则

Codex 必须根据原 `data.yaml` 动态生成映射。

目标类别固定为：

```text
crack → 0
water → 1
leak → 2
no leak → 删除该框
```

名称匹配应兼容：

```text
no leak
no_leak
noleak
No Leak
NO LEAK
```

### 7.2 标签转换

每一行原始标签：

```text
class_id x_center y_center width height
```

转换时：

1. 查原始类别名；
2. 若为 `no leak`，跳过；
3. 若为 `crack/water/leak`，重映射类别 ID；
4. 保留归一化坐标；
5. 若图片最终无目标，创建空 `.txt`。

### 7.3 自动质量检查

自动执行：

- 完全重复图片 SHA256 检查；
- 跨 train/valid/test 重复图片检查；
- 无效 BBox 删除；
- 宽或高小于等于 0 的框删除；
- 面积超过整图 95% 的框标记为可疑；
- 极小框面积小于 0.0001 的框标记为可疑；
- 图像打不开时跳过并记录；
- 标签类别越界时跳过并记录。

输出：

```text
models/metrics/preprocess_report.json
models/metrics/suspicious_samples.csv
```

POC 阶段不要求人工逐张确认，但必须保留异常样本列表。

---

## 8. 第三阶段：数据审计

创建：

```text
training/audit_dataset.py
```

必须验证：

1. `data/processed/data.yaml` 存在；
2. 类别固定为 `crack、water、leak`；
3. 所有标签类别 ID 只允许 `0、1、2`；
4. 所有坐标合法；
5. 图片与标签匹配；
6. train、valid、test 之间无完全相同图片；
7. 输出每类样本分布；
8. 输出负样本数量；
9. 输出每张图片目标数量分布。

若存在严重错误，应退出并返回非 0 状态码，禁止开始训练。

---

## 9. 第四阶段：完整微调训练

创建：

```text
training/train_full.py
```

默认配置：

```python
MODEL = "yolo11s.pt"
EPOCHS = 150
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0
WORKERS = 0
OPTIMIZER = "AdamW"
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
PATIENCE = 30
SEED = 42
AMP = True
```

要求：

- Windows 默认 `workers=0`；
- 显存不足时允许通过 CLI 或配置文件降低 batch；
- 参数从 `training/configs/train.yaml` 读取；
- 支持命令行覆盖；
- 训练结果写入：

```text
models/trained/yolo11s_full/
```

最终模型：

```text
models/trained/yolo11s_full/weights/best.pt
```

---

## 10. 第五阶段：LoRA 微调训练

创建：

```text
training/train_lora.py
```

使用 YOLO-Master 支持的 LoRA 参数。

默认配置：

```python
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
lora_gradient_checkpointing = True
```

其他训练参数与完整微调保持一致。

输出：

```text
models/trained/yolo11s_lora/
```

最终模型：

```text
models/trained/yolo11s_lora/weights/best.pt
```

若当前 YOLO-Master API 不一致：

1. 先检查实际可用的 `model.train()` 参数；
2. 不允许伪造 LoRA 已成功；
3. 在 README 中记录适配方式；
4. 若暂时无法完成 LoRA，完整微调仍需可运行；
5. 后端必须能加载任意兼容的 `best.pt`。

---

## 11. 第六阶段：模型评估

### 11.1 内部测试

创建：

```text
training/evaluate_internal.py
```

使用：

```text
data/processed/test
```

输出：

```text
Precision
Recall
mAP50
mAP50-95
每类 mAP50-95
每类目标数量
推理时间
混淆矩阵
PR 曲线
F1 曲线
```

保存：

```text
models/metrics/full_internal_metrics.json
models/metrics/lora_internal_metrics.json
```

### 11.2 模型比较

创建：

```text
training/compare_models.py
```

排序规则：

1. Recall 更高优先；
2. Recall 差异小于 0.02 时，mAP50 更高优先；
3. 再比较 mAP50-95；
4. 再比较推理时间；
5. 再比较模型文件大小。

输出：

```text
models/metrics/model_comparison.json
models/metrics/model_comparison.csv
models/metrics/selected_model.json
```

禁止写死假指标。

---

## 12. 外部测试支持

创建：

```text
training/evaluate_external.py
```

外部数据目录：

```text
data/external_test/
```

外部数据必须使用与训练模型一致的类别顺序：

```yaml
0: crack
1: water
2: leak
```

如果外部数据只包含 `leak`，其标签仍必须使用类别 ID `2`。

若外部数据没有标签：

- 只生成定性检测结果；
- 不计算 Precision、Recall、mAP；
- 页面标记为“外部图片定性测试”。

---

## 13. 模型部署准备

创建：

```text
training/export_model.py
```

功能：

1. 读取 `models/metrics/selected_model.json`；
2. 将选择的模型复制到 `models/deployed/best.pt`；
3. 尝试导出 `models/deployed/best.onnx`；
4. 环境支持时尝试导出 TensorRT；
5. 导出失败不能影响 `.pt` 推理；
6. 保存 `models/deployed/model_info.json`。

---

## 14. 后端 API

FastAPI 默认地址：

```text
http://127.0.0.1:8000
```

API 前缀：

```text
/api/v1
```

### 14.1 健康检查

```http
GET /api/v1/health
```

### 14.2 图片检测

```http
POST /api/v1/detect/image
```

输入：

```text
multipart/form-data
file
confidence
iou
use_llm
```

返回字段至少包括：

```text
record_id
status
status_zh
risk_level
risk_level_zh
counts
detections
summary
advice
original_image_url
result_image_url
report_url
```

每条检测结果至少包括：

```text
class_id
class_name
class_name_zh
confidence
bbox_xyxy
bbox_normalized
area_ratio
location_text
```

### 14.3 模型指标

```http
GET /api/v1/metrics
```

返回完整微调、LoRA、选中模型、每类指标、推理时间和模型大小。

### 14.4 历史记录

```http
GET /api/v1/history
GET /api/v1/history/{record_id}
DELETE /api/v1/history/{record_id}
```

### 14.5 报告

```http
GET /api/v1/reports/{record_id}
```

返回可打印 HTML。

### 14.6 视频检测

```http
POST /api/v1/detect/video
```

要求：

- 支持 mp4；
- 按固定帧间隔或逐帧检测；
- 生成带框视频；
- 输出检测总数；
- 输出出现泄漏的时间区间；
- 返回结果视频 URL。

---

## 15. YOLO 推理服务

创建：

```text
backend/services/yolo_service.py
```

要求：

1. 应用启动时只加载一次模型；
2. 自动选择 CUDA，否则使用 CPU；
3. 支持置信度和 IoU 阈值；
4. 输出结构化检测结果；
5. 绘制检测框；
6. 类别颜色保持一致；
7. 不允许每次请求重新加载模型；
8. 对损坏图片、非图片文件、超大文件进行异常处理；
9. 模型路径来自环境变量：

```text
YOLO_MODEL_PATH=models/deployed/best.pt
```

---

## 16. 综合状态判断

```python
if leak_count > 0:
    status = "suspected_leak"
    status_zh = "检测到疑似泄漏"
elif crack_count > 0 or water_count > 0:
    status = "related_anomaly"
    status_zh = "未检测到明确泄漏，但发现相关异常"
else:
    status = "no_visible_anomaly"
    status_zh = "当前图像中未检测到明显异常"
```

禁止输出：

```text
确认无泄漏
绝对安全
管道完全正常
```

---

## 17. 风险等级规则

创建：

```text
backend/services/risk_service.py
```

高风险满足任一条件：

```text
leak 置信度 >= 0.75
leak 数量 >= 2
leak 框面积比例 >= 0.05
leak 与 crack 的框 IoU >= 0.10
```

中风险：

```text
检测到 leak，但最高置信度 < 0.75
检测到 crack
检测到 water
```

无明显异常：

```text
未检测到 leak、crack、water
```

规则放在：

```text
backend/config/risk_rules.yaml
```

---

## 18. 位置描述

根据 BBox 中心坐标生成：

```text
左上
上部中央
右上
左侧中央
图像中央
右侧中央
左下
下部中央
右下
```

前端示例：

```text
疑似泄漏位于图像中部偏左
```

不得把图像位置描述成真实设备部件位置，除非模型增加了设备部件类别。

---

## 19. 建议生成

创建：

```text
backend/services/advice_service.py
```

### 19.1 固定规则

#### leak

```text
设置现场警戒区域，避免无关人员接近；
确认泄漏介质、压力状态和附近点火源；
检查泄漏区域附近的接头、法兰、阀门和密封件；
结合压力、流量和人工现场检查确认泄漏源；
不得仅依据图像直接拆卸或处置带压设备。
```

#### crack

```text
检查裂纹长度、宽度及是否继续扩展；
安排超声测厚、渗透检测或其他无损检测；
结合设计压力和剩余壁厚评估是否需要降压、修补或更换。
```

#### water

```text
排查水体来源；
判断水体是否与管道、接头或周边排水有关；
检查是否存在持续滴流、喷射或积水扩展；
避免把普通积水直接判断为管道泄漏。
```

#### 无明显异常

```text
当前图像未发现模型可识别的明显异常；
仍需按照既定巡检规程检查压力、流量、设备状态和不可见区域。
```

### 19.2 本地大语言模型

创建：

```text
backend/services/llm_service.py
```

配置：

```text
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
ENABLE_LLM=false
```

LLM 输入只能包括：

```text
检测类别
置信度
数量
位置描述
风险等级
固定规则建议
限制条件
```

提示词必须约束：

```text
不得补充未检测到的故障；
不得确认内部状态；
不得确定精确泄漏原因；
不得替代人工和传感器检查；
只润色建议；
输出简洁中文。
```

Ollama 不可用时自动回退固定规则。

---

## 20. 数据库

使用 SQLite。

表：

```text
detection_records
```

字段建议：

```text
id
created_at
source_type
original_filename
original_path
result_path
status
risk_level
confidence_threshold
iou_threshold
detections_json
summary
advice
model_version
inference_ms
video_result_path
duration
frame_count
leak_time_ranges_json
```

---

## 21. 前端页面

### 21.1 总览页 `/`

展示：

- 系统名称；
- 当前部署模型；
- GPU/CPU 状态；
- 最近检测数量；
- 疑似泄漏数量；
- 裂纹数量；
- 可见水体数量；
- 当前模型 Precision、Recall、mAP50、mAP50-95；
- 最近检测记录。

标题：

```text
管道泄漏多目标智能检测系统
```

副标题：

```text
基于 YOLO 的裂纹、可见水体与疑似泄漏定位
```

### 21.2 图片检测页 `/detect/image`

要求：

- 拖拽上传；
- 支持 JPG、JPEG、PNG、WEBP；
- 置信度滑块；
- IoU 滑块；
- 是否启用本地大模型；
- 开始检测；
- 加载状态；
- 原图/检测图左右对比；
- 可选前后对比滑杆；
- 检测详情表；
- 综合状态；
- 风险等级；
- 处理建议；
- 保存结果图；
- 打开报告；
- 重新检测。

### 21.3 视频检测页 `/detect/video`

要求：

- 上传 mp4；
- 显示处理进度；
- 播放原视频和结果视频；
- 显示异常时间段；
- 显示每类目标次数；
- 下载结果视频。

### 21.4 模型指标页 `/metrics`

展示：

- 完整微调 vs LoRA；
- Precision；
- Recall；
- mAP50；
- mAP50-95；
- 推理时间；
- 模型大小；
- 每类指标；
- PR/F1/混淆矩阵；
- 明确标注“内部测试集”。

不得显示虚构指标。无指标文件时显示“尚未完成模型评估”。

### 21.5 历史记录页 `/history`

功能：

- 表格；
- 按日期、状态、风险等级筛选；
- 查看详情；
- 打开报告；
- 删除记录；
- 显示原图和结果图缩略图。

### 21.6 报告页 `/reports/:id`

内容：

- 报告编号；
- 检测时间；
- 模型版本；
- 原始图片；
- 检测结果图片；
- 检测目标详情；
- 综合状态；
- 风险等级；
- 建议；
- 模型局限性；
- 打印按钮。

---

## 22. 页面视觉要求

采用企业级工业风格：

```text
深蓝/灰白为主
红色表示高风险
橙色表示中风险
绿色表示未发现明显异常
```

要求：

- 页面整洁；
- 不使用过多动画；
- 不暴露训练代码；
- 中文为主；
- 支持 1920×1080；
- 兼容 Chrome；
- 检测框类别颜色固定。

---

## 23. 固定免责声明

所有检测结果和报告必须包含：

> 本结果仅基于可见光图像中的外部视觉特征生成，不能直接确认泄漏介质、内部腐蚀、内部堵塞或精确故障原因，不能替代压力、流量、可燃气体、无损检测及现场人工安全检查。

---

## 24. 配置文件

创建 `.env.example`：

```dotenv
APP_NAME=管道泄漏多目标智能检测系统
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
FRONTEND_PORT=5173

YOLO_MODEL_PATH=models/deployed/best.pt
YOLO_DEVICE=0
DEFAULT_CONFIDENCE=0.25
DEFAULT_IOU=0.50
MAX_IMAGE_SIZE_MB=20
MAX_VIDEO_SIZE_MB=500

DATABASE_URL=sqlite:///storage/database/app.db

ENABLE_LLM=false
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
```

---

## 25. PowerShell 脚本

### `scripts/setup_windows.ps1`

- 检查 Python；
- 创建虚拟环境；
- 安装后端依赖；
- 安装前端依赖；
- 创建目录；
- 复制 `.env.example` 为 `.env`；
- 检查 CUDA；
- 不覆盖已有环境。

### `scripts/train_all.ps1`

顺序：

```text
inspect_dataset.py
preprocess_dataset.py
audit_dataset.py
train_full.py
train_lora.py
evaluate_internal.py
compare_models.py
export_model.py
```

任一步失败应停止。

### 启动脚本

```text
scripts/start_backend.ps1
scripts/start_frontend.ps1
scripts/start_all.ps1
```

`start_all.ps1` 要求：

- 分别启动后端和前端；
- 等待后端健康检查通过；
- 再启动前端；
- 自动打开浏览器。

---

## 26. 测试要求

后端至少测试：

```text
健康检查
无模型文件
正常图片上传
非法文件
空文件
超大文件
无目标图片
多目标图片
建议规则
风险等级
历史记录
报告生成
```

数据脚本至少测试：

```text
不同类别顺序映射
no leak 删除
空标签生成
坐标越界
重复图片
缺失标签
```

前端必须保证：

```text
npm run build
```

成功。

---

## 27. 验收标准

### 数据处理

- 自动读取原始类别；
- 正确删除 `no leak` 框；
- 正常图片保留为空标签负样本；
- 最终类别为 `crack/water/leak`；
- 审计无严重错误。

### 模型

- 完整微调可运行；
- LoRA 尽可能完成；
- 生成真实 `best.pt`；
- 输出真实测试指标；
- 不写死或伪造指标。

### 后端

- 模型只加载一次；
- 图片检测成功；
- 返回结构化 JSON；
- 保存原图和结果图；
- 保存历史记录；
- 生成报告。

### 前端

- 页面完整；
- 上传图片可用；
- 原图/结果图可见；
- 多个检测框可见；
- 类别、置信度和数量可见；
- 风险和建议可见；
- 指标页、历史记录和报告可用。

### 安全表达

- 无目标时显示“未检测到明显异常”；
- 不显示“确认无泄漏”；
- 不把 `water` 直接等同于泄漏；
- 报告包含免责声明。

---

## 28. Codex 实施顺序

```text
1. 创建项目结构；
2. 完成数据检查；
3. 完成数据预处理；
4. 完成数据审计；
5. 完成完整微调脚本；
6. 完成 LoRA 脚本；
7. 完成评估和模型选择；
8. 完成模型部署服务；
9. 完成图片检测 API；
10. 完成风险与建议模块；
11. 完成数据库和历史记录；
12. 完成报告；
13. 完成 Vue 前端；
14. 完成视频检测；
15. 完成测试；
16. 完成一键启动脚本；
17. 更新 README。
```

---

## 29. Codex 工作规则

1. 修改前检查已有项目；
2. 不删除用户已有文件；
3. 不覆盖原始数据集；
4. 处理后的数据写入 `data/processed`；
5. 所有路径使用 `pathlib.Path`；
6. 所有脚本包含异常处理；
7. Windows 默认 `workers=0`；
8. 不假设原类别 ID；
9. 不伪造训练结果；
10. 不写死本机绝对路径；
11. 配置通过 `.env` 或 YAML 修改；
12. 后端模型单例加载；
13. 前端不得用假检测数据冒充真实结果；
14. 若模型尚未训练，页面显示明确提示；
15. 每阶段完成后运行测试；
16. 最终说明已完成项、未完成项、启动命令和已知限制。

---

## 30. README 必须包含

```text
项目介绍
能力边界
目录结构
环境要求
数据集放置方式
数据预处理命令
完整微调命令
LoRA 微调命令
模型评估命令
模型导出命令
后端启动命令
前端启动命令
一键启动命令
Ollama 可选配置
常见错误
CUDA 检查方法
Windows workers 设置
客户演示步骤
免责声明
```

---

## 31. 最终客户演示顺序

```text
1. 打开系统首页；
2. 展示模型指标；
3. 上传正常管道图片；
4. 展示“当前图像未检测到明显异常”；
5. 上传泄漏图片；
6. 展示 leak BBox、置信度和建议；
7. 上传包含裂纹和水体的图片；
8. 展示多类别、多目标结果；
9. 打开检测报告；
10. 展示历史记录；
11. 说明后续可使用客户现场数据继续 LoRA 微调。
```

---

## 32. 最终业务定义

本模型的准确定位是：

> 基于可见光图像的管道裂纹、可见水体和疑似泄漏多目标检测模型。

当前模型不能直接完成：

```text
不可见天然气泄漏检测
内部堵塞检测
内部腐蚀检测
精确泄漏量测量
泄漏介质确认
正式工业安全验收
```

后续正式油田项目需要：

```text
客户现场 RGB 图片
热红外或 OGI 图像
压力与流量数据
声学数据
无损检测数据
专家复核结果
```

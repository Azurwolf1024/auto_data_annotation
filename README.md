# auto-annotator

配置驱动的**通用图像标注引擎**：调用本地 [Ollama](https://ollama.com) 视觉模型，通过运行时参数切换 **5 种标注任务**、输出 **4 种标注格式**。纯 Python，跨平台，核心逻辑可 import 复用。

## 特性

- **5 种任务**：`box` 检测框 / `point` 物体中心点 / `ocr` 文字识别 / `caption` 图像描述 / `tag` 多标签
- **4 种输出**：`json`（每图一个）/ `csv`（整批汇总）/ `yolo`（box、point）/ `coco`（box、point）
- **结构化保证**：用 Ollama `format` 参数传 JSON Schema 强制输出；解析失败自动重试 2 次（共 3 次），仍失败则跳过并记日志
- **坐标归一化**：全部坐标统一到 0~1；模型若返回像素值会自动按图片宽高归一化
- **断点续跑**：中途中断再跑一次即可继续，已完成的图片不会重复调用模型
- **并发预留**：`--workers N` 用 `ThreadPoolExecutor` 并发（默认 1，串行）

## 安装

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 确认本地视觉模型（输出中 Capabilities 含 vision 的模型可用）
ollama list
ollama show qwen3.5:4b

# 3. 确保 Ollama 服务在运行（默认 http://localhost:11434）
```

> 本机实测可用模型：`qwen3.5:4b`（Capabilities: vision ✓），代码默认即该模型。

## 快速开始

```bash
# 图像描述（最简命令）
python main.py --task-type caption --image-dir images --output-dir out --output-format json --model qwen3.5:4b
```

## 五种 task_type 用法

### 1. box —— 目标检测框（闭集，必须有 --targets）

```bash
python main.py --task-type box --targets "圆形,正方形,三角形" \
  --image-dir images --output-dir out_box --output-format json --model qwen3.5:4b
```

输出 `out_box/shapes.json`：

```json
{
  "task_type": "box",
  "image_path": "images\\shapes.png",
  "image": "shapes.png",
  "result": {
    "objects": [
      {"label": "圆形", "confidence": 0.95, "x_min": 0.094, "y_min": 0.167,
       "x_max": 0.406, "y_max": 0.583}
    ]
  }
}
```

同时支持 YOLO / COCO：

```bash
python main.py --task-type box --targets "圆形,正方形,三角形" --image-dir images --output-dir out_yolo --output-format yolo
python main.py --task-type box --targets "圆形,正方形,三角形" --image-dir images --output-dir out_coco --output-format coco
```

### 2. point —— 物体中心点（闭集）

```bash
python main.py --task-type point --targets "圆形,正方形,三角形" \
  --image-dir images --output-dir out_point --output-format json --model qwen3.5:4b
```

point 同样支持 YOLO（pose 格式）与 COCO（keypoints）：

```bash
python main.py --task-type point --targets "圆形,正方形,三角形" --image-dir images --output-dir out_point_yolo --output-format yolo
python main.py --task-type point --targets "圆形,正方形,三角形" --image-dir images --output-dir out_point_coco --output-format coco
```

`out_point_yolo/labels/shapes.txt`（每行 `class cx cy w h kp_x kp_y kp_flag`，point 只有中心点故 `w=h=0`、`kp_flag=2`）：

```
0 0.406250 0.562500 0.000000 0.000000 0.406250 0.562500 2
1 0.906250 0.562500 0.000000 0.000000 0.906250 0.562500 2
```

### 3. ocr —— 文字识别（不使用 targets）

```bash
python main.py --task-type ocr --image-dir images --output-dir out_ocr \
  --output-format csv --model qwen3.5:4b
```

### 4. caption —— 图像描述

```bash
# 描述全图
python main.py --task-type caption --image-dir images --output-dir out_cap --output-format json

# 围绕指定对象描述（--targets 可选）
python main.py --task-type caption --targets "红色圆形" --image-dir images --output-dir out_cap2 --output-format csv
```

### 5. tag —— 多标签（候选池，输出 5~10 个）

```bash
python main.py --task-type tag --targets "室内,室外,几何,文字,简约,彩色" \
  --image-dir images --output-dir out_tag --output-format json
```

## 参数表

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--task-type` | 是 | - | `box` / `point` / `ocr` / `caption` / `tag` |
| `--image-dir` | 是 | - | 输入图片目录（递归扫描 jpg/jpeg/png/webp） |
| `--output-dir` | 是 | - | 输出目录，不存在会自动创建 |
| `--targets` | box/point 必填 | 空 | 逗号分隔候选标签 |
| `--output-format` | 否 | `json` | `json` / `csv` / `yolo` / `coco` |
| `--model` | 否 | `qwen3.5:4b` | Ollama 视觉模型 tag |
| `--ollama-url` | 否 | `http://localhost:11434` | Ollama 服务地址 |
| `--workers` | 否 | `1` | 并发线程数，>1 时用 ThreadPoolExecutor |
| `--confidence` | 否 | `0.3` | 置信度阈值，低于该值的项被丢弃（ocr 不过滤） |
| `--no-resume` | 否 | 关闭 | 关闭断点续跑，强制重跑全部图片 |

## 格式与任务兼容性

| output_format | box | point | ocr | caption | tag |
| --- | --- | :-: | :-: | :-: | :-: |
| `json` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `csv` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `yolo` | ✓ | ✓ | ✗ | ✗ | ✗ |
| `coco` | ✓ | ✓ | ✗ | ✗ | ✗ |

不兼容的组合会在启动阶段直接抛 `ValueError` 并退出（退出码 2）。

## 输出组织

```
json  -> output_dir/<stem>.json                 每图一个
yolo  -> output_dir/labels/<stem>.txt           每图一个
        output_dir/classes.txt                  类别名，行号即 class_index
csv   -> output_dir/annotations.csv             整批汇总（utf-8-sig）
coco  -> output_dir/annotations.json            整批汇总（标准 COCO）
```

YOLO 行格式（全部为归一化值，6 位小数）：

| task_type | 行格式 |
| --- | --- |
| box | `class_index x_center y_center width height` |
| point | `class_index x_center y_center width height kp_x kp_y kp_flag`（YOLO-pose；`w=h=0`，`kp_x=kp_y=中心点`，`kp_flag=2`） |

CSV 列定义：

| task_type | 列 |
| --- | --- |
| box | image,label,x1,y1,x2,y2,confidence |
| point | image,label,x,y,confidence |
| ocr | image,text,type,x1,y1,x2,y2 |
| caption | image,caption |
| tag | image,label,confidence |

## 目录结构

```
auto-annotator/
├── requirements.txt
├── README.md
├── main.py                 薄 CLI 封装
└── src/
    ├── __init__.py         包出口：Config / run / export
    ├── config.py           Config 数据类与校验
    ├── schema.py           pydantic v2 数据契约
    ├── prompts.py          五种中文 prompt
    ├── ollama_client.py    Ollama 调用 + JSON Schema + 重试
    ├── validator.py        结果清洗（闭集过滤、置信度、坐标裁剪）
    ├── pipeline.py         扫描 / 续跑 / 并发 / 进度
    └── exporters/
        ├── __init__.py     分发 + read_image_size + processed 状态
        ├── json_exporter.py
        ├── csv_exporter.py
        ├── yolo_exporter.py
        └── coco_exporter.py
```

## 作为库使用

```python
from src.config import Config
from src.pipeline import run
from src.exporters import export

config = Config(
    task_type="box",
    targets=["圆形", "正方形", "三角形"],
    image_dir="images",
    output_dir="out",
    output_format="coco",
    model_name="qwen3.5:4b",
).validate()

results = run(config)       # List[AnnotationResult]
info = export(results, config)
```

## 断点续跑

- `json` / `yolo`：检查同名输出文件是否存在（`<stem>.json`、`labels/<stem>.txt`）；
- `csv` / `coco`：维护 `output_dir/processed.json` 记录已处理图片的相对路径；
- 加 `--no-resume` 可强制全量重跑。

## 常见问题

**Q：报 `未安装 ollama Python 库`**
A：执行 `pip install -r requirements.txt`。

**Q：报连接错误 / `Connection refused`**
A：确认 Ollama 桌面端已启动，或用 `--ollama-url http://<host>:11434` 指向正确地址。

**Q：模型返回的内容解析失败，日志出现「标注失败（已尝试 3 次）」**
A：该图会被标记 `status="failed"` 并跳过，不影响其他图片。可换更大的视觉模型，或降低并发（`--workers 1`）后重试。

**Q：`yolo 格式只支持 ('box', 'point') 任务`**
A：YOLO 标签只有 bbox 与 pose 两种表达，`ocr` / `caption` / `tag` 请改用 `--output-format json` 或 `csv`。

**Q：坐标看起来不对**
A：代码会自动把 >1 的像素坐标按图片宽高归一化，并裁剪到 [0,1]；若仍异常，说明模型输出的框本身有偏差，可降低 `--confidence` 或换模型。

**Q：COCO 里 width/height 是 0**
A：图片尺寸由纯 Python 解析文件头获得，极少数损坏/特殊编码图片会解析失败；安装 Pillow 后会自动回退读取。

**Q：处理速度慢**
A：视觉模型推理本身耗时，可用 `--workers 4` 并发（注意显存/内存占用）。

## 许可

内部工具，按需修改使用。

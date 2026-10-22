# auto-annotator

配置驱动的**通用图像标注引擎**：调用本地 [Ollama](https://ollama.com) 视觉模型，通过运行时参数切换 **5 种标注任务**、输出 **4 种标注格式**。纯 Python，跨平台，核心逻辑可 import 复用。

## 特性

- **5 种任务**：`box` 检测框 / `point` 物体中心点 / `ocr` 文字识别 / `caption` 图像描述 / `tag` 多标签
- **4 种输出**：`json`（每图一个）/ `csv`（整批汇总）/ `yolo`（box、point）/ `coco`（box、point）
- **结构化保证**：用 Ollama `format` 参数传 JSON Schema 强制输出；解析失败自动重试 2 次（共 3 次），仍失败则跳过并记日志
- **坐标归一化**：全部坐标统一到 0~1；模型若返回像素值会自动按图片宽高归一化
- **point 可训练**：point 导出 YOLO/COCO 时自动把中心点膨胀成小方框（默认 32px），避免零面积框无法训练
- **断点续跑**：中途中断再跑一次即可继续，已完成的图片不会重复调用模型
- **二次验证（harness loop）**：`--verify` 开启后，box/point 检测出的每个候选会单独裁切出来再问一次模型「这里真有目标本体吗」，可挡掉 HUD / 文字横幅 / 武器视图等误检，并回贴修正框
- **模型主动放大（MCP 裁剪放大服务）**：`--zoom-on-uncertain` 开启后，模型可在结构化输出里主动声明 `zoom_regions`（--zoom-mode model，默认），harness 随即调用裁剪放大工具、重新标注并把坐标映射回原图；裁切图里还能再次声明，实现**多级递归放大**。也可退回旧的低置信触发（--zoom-mode threshold）。放大能力来自一个零依赖的轻量 MCP 服务器（`mcp_server/crop_zoom_server.py`），可被任意 MCP 宿主复用
- **可视化质检**：`--visualize` 在标注完成后把最终结果（框 / 点 / 文字框）画回原图，输出 `_viz/<stem>.png`，肉眼即可核对标注质量
- **并发预留**：`--workers N` 用 `ThreadPoolExecutor` 并发（默认 1，串行）；放大客户端整批复用且线程安全

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

`out_point_yolo/labels/shapes.txt`（每行 `class cx cy w h kp_x kp_y kp_flag`，`kp_flag=2`）：

```
0 0.406250 0.562500 0.050000 0.066667 0.406250 0.562500 2
1 0.906250 0.562500 0.050000 0.066667 0.906250 0.562500 2
```

> **为什么 w/h 不是 0**：零面积框在 YOLO 里学不到东西，因此 point 默认会把中心点
> **膨胀成边长 32 像素的正方形框**（640×480 图上即 `0.05 × 0.0667`）。
> `kp_x/kp_y` 始终是模型给出的原始中心点；靠边的点会把框向内收，保证框完整落在图内。
> 想改大小用 `--point-box-size`，传 `0` 则退回零面积框。COCO 的 `bbox`/`area` 同样按此规则生成。

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

### 6. 二次验证（harness loop，`--verify`）

box / point 检测经常把 HUD、文字横幅、武器视图、击杀飘字误判成目标。
开启 `--verify` 后，每个候选框会被单独裁切出来（按 `--verify-margin` 向外扩边），
再喂给模型问一次「这张裁切图里**真的**有完整可见的目标本体吗？」：

- 模型说 `present=false`，或返回的类别不在候选列表，或复核置信度 `< --verify-threshold` → **剔除**该候选；
- 模型说 `present=true` → **接受**，并把给出的修正框反投影回全图坐标（多轮时逐轮收敛）；
- 裁切图会存到 `output_dir/_verify/<stem>/candNN_rR.png`，方便你回看模型到底看到了什么。

```bash
# 游戏截图检测：挡掉 HUD / 击杀飘字等误检
python main.py --task-type box --targets "敌人,队友,载具" \
  --image-dir screens --output-dir out_box_v --output-format json \
  --verify --verify-domain-hints "游戏截图，注意区分真实角色与界面HUD/血条/击杀提示" \
  --verify-threshold 0.6

# 多轮收敛：先用初框裁切，再用修正框再裁切再问，最多 2 轮
python main.py --task-type point --targets "红色圆形,蓝色方块" \
  --image-dir images --output-dir out_point_v --output-format json \
  --verify --verify-rounds 2 --verify-margin 0.3
```

> **提示**：二次验证对每张候选多调用一次模型，会显著增加耗时与 token 消耗；
> 对误检不敏感的场景可不开。仅 box / point 支持，其他任务类型组合会直接报错退出。

### 7. 放大精修（MCP 裁剪放大服务，`--zoom-on-uncertain`）

小模型对远处/细小的目标经常「看不准」——框偏了、置信度低。开启
`--zoom-on-uncertain` 后，流水线会自动裁切放大、重新标注并把结果映射回原图。

#### 两种触发方式（`--zoom-mode`）

| `--zoom-mode` | 触发来源 | 说明 |
| --- | --- | --- |
| `model`（**默认**） | 模型在结构化输出里声明 `zoom_regions` | **模型主动 call 裁剪放大工具**：它自己判断哪里看不清，请求放大 |
| `threshold` | 置信度低于 `--zoom-threshold` 的候选 | 旧行为：机械地对低置信候选放大 |
| `both` | 两者都触发 | 兼顾 |

`model` 模式下，box / point 的 JSON schema 额外带一个 `zoom_regions` 字段，
模型可以像下面这样主动申请放大（可声明多个，也可以不声明 = 不需要放大）：

```json
{
  "objects": [
    {"label": "圆形", "confidence": 0.85,
     "x_min": 0.05, "y_min": 0.05, "x_max": 0.15, "y_max": 0.15}
  ],
  "zoom_regions": [
    {"x_min": 0.4, "y_min": 0.3, "x_max": 0.6, "y_max": 0.5,
     "reason": "目标太小，需要放大确认"}
  ]
}
```

#### 处理流程

对每个种子区域：

1. 以该区域为局部矩形，调用 **MCP 裁剪放大服务** 裁切并放大，得到更大的局部图；
2. 把局部图重新喂给检测模型；
3. **模型可以在裁切图里再次声明 `zoom_regions`**，从而逐级放大更深一层
   （坐标随每一级 `region` 一路 compose 回原图），直到 `--zoom-max-levels` 级；
4. 用最终的 `region` 把局部坐标**映射回原图坐标**；
5. 与已有检测按 IoU（point 按距离）合并：重合的同标签检测用精修结果替换，
   新发现的目标则追加。

每级的裁切图都会存到 `output_dir/_zoom/<stem>/`，方便你回看放大的过程。

```bash
# 默认 model 模式：由模型主动决定放大哪里
python main.py --task-type box --targets "圆形,正方形,三角形" \
  --image-dir images --output-dir out_zoom --output-format json \
  --zoom-on-uncertain --zoom-upscale 3 --zoom-max-levels 3

# 退回旧的低置信触发（阈值 0.5）
python main.py ... --zoom-on-uncertain --zoom-mode threshold --zoom-threshold 0.5

# 两者都要
python main.py ... --zoom-on-uncertain --zoom-mode both --zoom-threshold 0.5

# 关闭 MCP 子进程、改用进程内本地引擎（更快，无 stdio 握手）
python main.py ... --zoom-on-uncertain --no-zoom-mcp
```

> **与 `--verify` 的区别**：`--verify` 是「判断是否误检并剔除/修正」，放大精修是
> 「看得不够清就放大重看、提升定位精度」。两者可叠加使用（先 verify 剔除误检，
> 再对保留的候选放大精修）。

#### 独立运行 MCP 服务器

`mcp_server/crop_zoom_server.py` 是一个**零额外依赖**的轻量 MCP 服务
（仅用标准库，通过 stdio 通信），可被任意 MCP 宿主（如 Claude Desktop）直接挂接：

```bash
python -m mcp_server.crop_zoom_server
# 或
python mcp_server/crop_zoom_server.py
```

它暴露的工具：

| 工具 | 作用 |
| --- | --- |
| `crop_and_upscale` | 把图片某归一化区域裁切并放大，返回裁切图路径与 compose 后的原图矩形 `region` |
| `map_box_to_original` | 把相对某裁切图的归一化框映射回原图坐标 |
| `map_point_to_original` | 把相对某裁切图的点映射回原图坐标 |
| `health` | 健康检查 |

> `region` / `local` 三种写法都接受：`{"x_min":…}` 对象、长度 4 的数组
> `[x_min,y_min,x_max,y_max]`、以及带同名属性的对象。

典型多级放大调用序列（以原图 `region={0,0,1,1}` 为例）：

```
# 第 1 级：在原图里裁切 (0.2,0.2)-(0.6,0.6) 放大 2 倍
crop_and_upscale(image, {0,0,1,1}, [0.2,0.2,0.6,0.6], 2)
  -> {cropped_image_path, region: {0.2,0.2,0.6,0.6}, size:[...]}
# 第 2 级：在上一级裁切图里再裁切 (0.3,0.3)-(0.7,0.7) 放大 2 倍
crop_and_upscale(crop1, {0.2,0.2,0.6,0.6}, [0.3,0.3,0.7,0.7], 2)
  -> region: {0.32,0.32,0.44,0.44}   # 已 compose 回原图
# 把局部检测框映射回原图
map_box_to_original({0.32,0.32,0.44,0.44}, [lx,ly,rx,ry]) -> 原图坐标
```

### 8. 可视化标注图（`--visualize`）

想肉眼检查标注质量时，加 `--visualize`：流水线会在**标注完成后**把最终结果
画回原图，输出 `<viz-dir>/<图片名>.png`（默认 `<output-dir>/_viz/`）。

- **box**：矩形框 + 标签 + 置信度文字；
- **point**：圆点 + 十字准星 + 标签；
- **ocr**：文字块矩形 + 文字片段；
- 同标签稳定同色，文字带黑底，保证在亮/暗图上都可读；
- caption / tag 没有几何信息，仅复制原图以便对照。

画的永远是**最终坐标**——即经过 verify 剔除和放大精修映射回原图之后的结果，
所以看到的就是导出数据里的坐标。

```bash
python main.py --task-type box --targets "圆形,正方形" \
  --image-dir images --output-dir out --output-format yolo \
  --zoom-on-uncertain --verify --visualize

# 指定可视化目录
python main.py ... --visualize --viz-dir out/preview
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
| `--point-box-size` | 否 | `32` | point 导出 yolo/coco 时中心点膨胀成的方框边长（像素），`0` = 零面积框 |
| `--no-resume` | 否 | 关闭 | 关闭断点续跑，强制重跑全部图片 |
| `--verify` | 否 | 关闭 | 开启二次验证（harness loop），仅 box/point 有效 |
| `--verify-rounds` | 否 | `1` | 验证轮数；每轮用上一轮修正框再裁切再问，直到稳定或轮次耗尽 |
| `--verify-margin` | 否 | `0.25` | 验证时向外扩边比例（相对框尺寸）；point 任务视为裁切半边长（归一化） |
| `--verify-threshold` | 否 | `0.5` | 复核置信度阈值，低于则剔除该候选 |
| `--verify-domain-hints` | 否 | 空 | 领域线索写入验证 prompt（如「游戏截图」「医学影像」），帮助模型识别误检 |
| `--zoom-on-uncertain` | 否 | 关闭 | 开启放大精修：裁切放大后重新标注并映射回原图，仅 box/point 有效 |
| `--zoom-mode` | 否 | `model` | 触发方式：`model`=模型主动请求的 `zoom_regions`（默认）/ `threshold`=低置信触发 / `both` |
| `--zoom-upscale` | 否 | `2` | 放大倍数（像素维度） |
| `--zoom-threshold` | 否 | `0.3` | 触发放大的置信度阈值，低于则放大精修 |
| `--zoom-max-levels` | 否 | `2` | 最大放大级数（支持多级/递归放大） |
| `--zoom-margin` | 否 | `0.25` | point 任务每次放大的裁切半边长（归一化） |
| `--no-zoom-mcp` | 否 | 关闭 | 关闭通过 MCP 服务调用放大，改用进程内本地引擎（无子进程） |
| `--zoom-server` | 否 | 自动 | MCP 服务器脚本路径，默认 `mcp_server/crop_zoom_server.py` |
| `--visualize` | 否 | 关闭 | 把最终标注（框/点/文字框）画回原图，生成 `<viz-dir>/<stem>.png` 供人工检查 |
| `--viz-dir` | 否 | `<output-dir>/_viz` | 可视化输出目录 |

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
_verify -> output_dir/_verify/<stem>/candNN_rR.png  仅 --verify 时生成，每张候选的复核裁切图（供人工回看）
_zoom   -> output_dir/_zoom/<stem>/crop_sN_Lk.png   仅 --zoom-on-uncertain 时生成，每级放大的裁切图
_viz    -> output_dir/_viz/<stem>.png               仅 --visualize 时生成，带框/点的标注预览图
```

> `_verify/` 与 `_zoom/` 都是中间产物，仅供你肉眼复核「模型究竟看到了什么 / 放大到了什么」；
> `_viz/` 是最终标注的可视化预览。三者都可随时删除，不影响导出数据。

YOLO 行格式（全部为归一化值，6 位小数）：

| task_type | 行格式 |
| --- | --- |
| box | `class_index x_center y_center width height` |
| point | `class_index x_center y_center width height kp_x kp_y kp_flag`（YOLO-pose；`w/h` 为中心点膨胀框，默认 32px，`kp_x/kp_y` 为原始中心点，`kp_flag=2`） |

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
├── mcp_server/
│   ├── __init__.py
│   └── crop_zoom_server.py 轻量 MCP 服务（裁剪/放大 + 坐标映射，零额外依赖）
└── src/
    ├── __init__.py         包出口：Config / run / export
    ├── config.py           Config 数据类与校验
    ├── schema.py           pydantic v2 数据契约
    ├── prompts.py          五种中文 prompt
    ├── ollama_client.py    Ollama 调用 + JSON Schema + 重试
    ├── validator.py        结果清洗（闭集过滤、置信度、坐标裁剪）
    ├── mcp_zoom.py         放大精修客户端（MCP stdio 客户端 / 本地引擎）
    ├── visualize.py        把标注结果画回原图（框 / 点 / 文字框）
    ├── pipeline.py         扫描 / 续跑 / 并发 / 进度 / 验证 / 放大精修 / 可视化
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
A：代码会自动把 >1 的像素坐标按图片宽高归一化，并裁剪到 [0,1]（`zoom_regions` 同样会换算）；若仍异常，说明模型输出的框本身有偏差，可降低 `--confidence` 或换模型。

**Q：开了 `--zoom-on-uncertain` 但 `_zoom/` 是空的**
A：默认 `--zoom-mode model`，只有模型主动声明 `zoom_regions` 才会放大。想强制对低置信候选放大，改用 `--zoom-mode threshold`（或 `both`）；也可调高 `--zoom-threshold`。

**Q：point 导出的框太大/太小**
A：默认 32 像素边长，用 `--point-box-size` 调；传 `0` 得到零面积框（仅 pose 关键点，不能用于检测训练）。

**Q：COCO 里 width/height 是 0**
A：图片尺寸由纯 Python 解析文件头获得，极少数损坏/特殊编码图片会解析失败；安装 Pillow 后会自动回退读取。

**Q：`--verify 仅支持 box / point 任务`**
A：二次验证是对检测候选做裁切复核，只有 box / point 有「候选位置」可裁切；ocr / caption / tag 请用不带 `--verify` 的普通流程。

**Q：`--zoom-on-uncertain 仅支持 box / point 任务`**
A：放大精修依赖候选的「位置」去裁切局部图，只有 box / point 有位置信息；ocr / caption / tag 暂不支持。

**Q：开启 `--zoom-on-uncertain` 后输出目录多了 `_zoom/`**
A：那是每张低置信度候选的逐级放大裁切图（供你回看放大到了什么），纯中间产物，可随时删除，不影响最终 `json` / `yolo` / `coco` 输出。

**Q：放大精修没生效 / 没有候选被放大**
A：只有置信度 `< --zoom-threshold` 的候选才会被放大；若模型给的置信度都很高则不会触发。可把 `--zoom-threshold` 调高（如 0.6）来覆盖更多候选。

**Q：想把这个裁剪放大能力用在自己的 Agent / Claude Desktop 里**
A：直接运行 `python -m mcp_server.crop_zoom_server`，它是一个零依赖的 MCP stdio 服务，暴露 `crop_and_upscale` / `map_box_to_original` / `map_point_to_original` / `health` 四个工具，可被任意 MCP 宿主挂接。

**Q：开启 `--verify` 后输出目录多了 `_verify/`**
A：那是每张候选的复核裁切图（供你回看模型看到了什么），纯中间产物，可随时删除，不影响最终 `json` / `yolo` / `coco` 输出。

**Q：二次验证把原本正确的框也剔除了**
A：优先排查 `--verify-threshold`（默认 0.5，调低更易接受）与 `--verify-domain-hints`（给模型更明确的误检界定）。若模型对裁切子图判断不稳，可减小 `--verify-margin` 让裁切更贴合目标本身。

**Q：处理速度慢**
A：视觉模型推理本身耗时，可用 `--workers 4` 并发（注意显存/内存占用）。

## 许可

内部工具，按需修改使用。

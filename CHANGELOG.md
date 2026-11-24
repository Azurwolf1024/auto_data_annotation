# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。版本号定义在 `src/__init__.py` 的 `__version__`。

## [0.2.0] - 2026-11-24

基于在本机 RTX 5060 Ti / `qwen3.5:4b` 上对高分辨率游戏截图（2945×1687）的实测，
修复了几类「看似跑了但结果为空/残缺/坐标全错」的隐性问题。

### 修复

- **大图撑爆上下文窗**：一张大图约占 4090 图像 token，超过模型默认 `num_ctx=4096`，
  导致输出被截断到 4~6 token。现在按图片面积自动估算 `num_ctx`，解析失败时用响应里
  真实的 `prompt_eval_count` 只升不降地重试扩容。（`--num-ctx`，默认 0=自适应）
- **think 逻辑反了**：原代码首次调用反而开启思考，实测思考模式下 token 全花在推理、
  结构化输出几乎为空。现在所有尝试都显式 `think:false`。
- **0~1000 归一化坐标误判**：原逻辑把 >1 的值一律当像素除以图宽，坐标缩小约 3 倍、
  框全落在左上角。新增 `detect_coord_scale()` 区分 unit / norm1000 / pixel 三种尺度，
  应用于 box / point / ocr / zoom_regions。
- **输出预算不足导致 JSON 截断**：密集 OCR 输出可达上千 token，默认 `num_predict`
  从 1024 提升到 2048，并在检测到顶到预算时翻倍重试。（`--max-tokens`）
- **JSON 解析容错**：模型偶尔返回单元素数组 `[{...}]` 或在 JSON 前后夹带文字，
  `parse_content` 现在会拆包数组、截取 JSON 主体。

### 新增

- CLI 参数 `--num-ctx`、`--max-tokens`。
- 回归测试 `tests/test_ollama_client_fixes.py`（13 项），覆盖坐标尺度识别、
  `num_ctx` 自适应、数组解析等修复；已纳入 `python -m tests` 统一入口。
- 文档：安装环境要求、测试章节、坐标 FAQ、贡献指南。

## [0.1.0] - 2026-10-22

配置驱动的通用图像标注引擎首个可用版本。

### 新增

- 5 种标注任务：`box` / `point` / `ocr` / `caption` / `tag`。
- 4 种输出格式：`json` / `csv` / `yolo` / `coco`；point 导出自动膨胀成小方框避免零面积框。
- JSON Schema 强制结构化输出 + 失败重试、闭集过滤与置信度阈值、坐标统一归一化到 0~1。
- 断点续跑：`json`/`yolo` 按文件存在判断，`csv`/`coco` 用 `processed.json` 记录。
- 二次验证 harness loop（`--verify`）：裁切候选图复核，剔除 HUD / 文字横幅等误检。
- 轻量 MCP 裁剪放大服务（`mcp_server/crop_zoom_server.py`）+ 多级递归放大精修
  （`--zoom-on-uncertain`，支持模型主动声明 `zoom_regions`）。
- 可视化质检（`--visualize`）、`--workers` 并发预留。
- 测试套件：MCP 协议层 + harness 集成层（`python -m tests`）。

## [0.0.1] - 2026-10-18

初始提交：项目骨架。

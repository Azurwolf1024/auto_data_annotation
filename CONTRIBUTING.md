# 贡献指南

## 开发环境

- Python 3.10+
- 依赖：`pip install -r requirements.txt`
- 可选：本地 Ollama 服务 + vision 模型（跑真实标注才需要，**测试不需要**）

建议在虚拟环境中开发：

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 运行测试

```bash
# 一条命令跑全部套件（MCP 协议层 + harness 集成 + Ollama 客户端修复）
python -m tests

# 单独跑某个套件
python tests/test_mcp_zoom.py
python tests/test_harness_zoom.py
python tests/test_ollama_client_fixes.py
```

测试全部用 mock，**不依赖 Ollama 服务或 GPU**，可离线运行。
测试入口 `tests/__main__.py` 会把 `tests/` 与项目根目录都加入 `sys.path`，
所以从任意目录调用也能正确 `import src...`。

新增测试请放进 `tests/test_*.py`，如果是独立套件，记得在 `tests/__main__.py`
的 `main()` 里注册，让 `python -m tests` 能一键覆盖。

## 代码结构

| 模块 | 职责 |
| --- | --- |
| `main.py` | CLI 参数解析，组装 `Config` 后调用 `run()` / `export()` |
| `src/config.py` | `Config` 数据类、校验、CLI/环境变量装载 |
| `src/schema.py` | pydantic v2 数据契约（各任务的返回值结构） |
| `src/prompts.py` | 各任务的 prompt 模板 |
| `src/ollama_client.py` | Ollama 调用、JSON Schema、`num_ctx` 自适应、重试与解析 |
| `src/validator.py` | 结果清洗：闭集过滤、置信度、坐标尺度识别与裁剪 |
| `src/pipeline.py` | 扫描 / 续跑 / 并发 / 验证 / 放大精修 / 可视化主流程 |
| `src/mcp_zoom.py` | 放大精修客户端（MCP stdio / 进程内本地引擎） |
| `src/visualize.py` | 把标注结果画回原图 |
| `src/exporters/` | 四种格式导出 |
| `mcp_server/` | 零依赖的裁剪放大 MCP 服务，可被外部宿主复用 |

## 改动约定

- **坐标一律用 0~1 归一化**：任何新增的模型输出解析，都要经过 `validator.detect_coord_scale()`
  处理，不要假设模型返回的一定是像素值或一定是 0~1000。
- **结构化输出必须 `think:false`**：实测思考模式会让结构化输出为空。
- **换模型/换任务前先实测**：本项目的不少参数（`num_ctx`、`num_predict`、坐标尺度）
  来自对 `qwen3.5:4b` 的实测，换模型后建议先用一张图跑一遍核对。
- **改动行为请同步更新**：`README.md`、`CHANGELOG.md`，并补上对应回归测试。

## 提交与发布

- 提交信息用英文祈使句，例如 `feat: ...` / `fix: ...` / `docs: ...`。
- 版本号更新在 `src/__init__.py` 的 `__version__`，并在 `CHANGELOG.md` 记录变更。

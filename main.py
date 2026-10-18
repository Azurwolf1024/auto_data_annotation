# -*- coding: utf-8 -*-
"""main.py —— 薄 CLI 封装。

业务逻辑全部在 src/ 中，本文件只负责：
解析参数 -> 构造并校验 Config -> 跑 pipeline -> 导出结果 -> 打印统计。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 允许直接 `python main.py` 运行（无需安装成包）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEFAULT_MODEL, OUTPUT_FORMATS, TASK_TYPES, Config  # noqa: E402
from src.exporters import export  # noqa: E402
from src.pipeline import run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-annotator",
        description="配置驱动的通用图像标注引擎（本地 Ollama 视觉模型）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=list(TASK_TYPES),
        help="标注任务类型：box 检测框 / point 中心点 / ocr 文字 / caption 描述 / tag 打标",
    )
    parser.add_argument(
        "--targets",
        default="",
        help="候选标签，逗号分隔。box/point 必填（闭集），tag 为候选池，caption 可选",
    )
    parser.add_argument("--image-dir", required=True, help="输入图片目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--output-format",
        default="json",
        choices=list(OUTPUT_FORMATS),
        help="输出格式：json（每图一个）/ csv（整批汇总）/ yolo（仅 box）/ coco（box/point）",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama 视觉模型名，默认 {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434", help="Ollama 服务地址"
    )
    parser.add_argument("--workers", type=int, default=1, help="并发线程数，默认 1（串行）")
    parser.add_argument(
        "--confidence", type=float, default=0.3, help="置信度阈值，低于该值的结果被丢弃"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="关闭断点续跑，重新处理所有图片",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = Config.from_args(args).validate()
    except ValueError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    os.makedirs(config.output_dir, exist_ok=True)
    print(f"输出目录：{config.output_dir}")

    try:
        results = run(config)
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130

    try:
        info = export(results, config)
    except ValueError as exc:
        print(f"[导出错误] {exc}", file=sys.stderr)
        return 3

    ok = sum(1 for r in results if r.status == "ok")
    failed = sum(1 for r in results if r.status != "ok")

    print("-" * 60)
    print(f"任务类型   ：{config.task_type}")
    print(f"输出格式   ：{config.output_format}")
    print(f"图片总数   ：{len(results)}")
    print(f"成功       ：{ok}")
    print(f"失败       ：{failed}")
    print(f"输出位置   ：{info.get('path')}")
    if failed:
        print("失败明细：")
        for item in results:
            if item.status != "ok":
                print(f"  - {Path(item.image_path).name}: {item.error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

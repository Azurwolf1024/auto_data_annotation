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

from src.config import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_POINT_BOX_SIZE,
    OUTPUT_FORMATS,
    TASK_TYPES,
    Config,
)
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
        help="输出格式：json（每图一个）/ csv（整批汇总）/ yolo（box、point）/ coco（box、point）",
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
        "--point-box-size",
        type=float,
        default=None,
        help=(
            f"point 任务导出 yolo/coco 时，把中心点膨胀成的方框边长（像素，默认 "
            f"{DEFAULT_POINT_BOX_SIZE:g}）；传 0 表示保持零面积框"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="关闭断点续跑，重新处理所有图片",
    )
    # ---- 二次验证（harness loop）----
    parser.add_argument(
        "--verify",
        action="store_true",
        help="开启二次验证：box/point 检测后再逐候选裁切复核，剔除误检（HUD/文字横幅/武器视图等）",
    )
    parser.add_argument(
        "--verify-rounds",
        type=int,
        default=1,
        help="验证轮数，默认 1；每轮用上一轮修正框再裁切再问，直到稳定或轮次耗尽",
    )
    parser.add_argument(
        "--verify-margin",
        type=float,
        default=0.25,
        help="验证时向外扩边比例（相对框尺寸）；point 任务视为裁切半边长（归一化），默认 0.25",
    )
    parser.add_argument(
        "--verify-threshold",
        type=float,
        default=0.5,
        help="复核置信度阈值，低于则剔除该候选，默认 0.5",
    )
    parser.add_argument(
        "--verify-domain-hints",
        default="",
        help="领域线索，写入验证 prompt（例如『游戏截图』『医学影像』），帮助模型识别误检",
    )
    # ---- 放大精修（zoom-on-uncertain）----
    parser.add_argument(
        "--zoom-on-uncertain",
        action="store_true",
        help="开启放大精修：box/point 中置信度低于阈值的候选，自动裁切放大后重新标注并映射回原图",
    )
    parser.add_argument(
        "--zoom-upscale",
        type=int,
        default=2,
        help="放大倍数（像素维度），默认 2",
    )
    parser.add_argument(
        "--zoom-threshold",
        type=float,
        default=None,
        help="触发放大的置信度阈值，低于则放大精修，默认 0.3",
    )
    parser.add_argument(
        "--zoom-max-levels",
        type=int,
        default=2,
        help="最大放大级数（支持多级/递归放大），默认 2",
    )
    parser.add_argument(
        "--zoom-margin",
        type=float,
        default=None,
        help="point 任务每次放大的裁切半边长（归一化），默认 0.25",
    )
    parser.add_argument(
        "--no-zoom-mcp",
        dest="zoom_via_mcp",
        action="store_false",
        help="关闭通过 MCP 服务调用放大（改用进程内本地引擎，无子进程）",
    )
    parser.add_argument(
        "--zoom-server",
        default="",
        help="MCP 服务器脚本路径；默认自动定位 mcp_server/crop_zoom_server.py",
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

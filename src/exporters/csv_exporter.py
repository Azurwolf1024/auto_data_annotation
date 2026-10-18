# -*- coding: utf-8 -*-
"""CSV 导出：整批汇总到 output_dir/annotations.csv（utf-8-sig 编码）。"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from . import relative_name, save_processed
from ..config import Config

COLUMNS = {
    "box": ["image", "label", "x1", "y1", "x2", "y2", "confidence"],
    "point": ["image", "label", "x", "y", "confidence"],
    "ocr": ["image", "text", "type", "x1", "y1", "x2", "y2"],
    "caption": ["image", "caption"],
    "tag": ["image", "label", "confidence"],
}

CSV_NAME = "annotations.csv"


def _fmt(value) -> str:
    """数字统一 6 位小数，去掉多余的 0。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    return f"{number:.6f}"


def _rows_of(item, config: Config):
    """把单个 AnnotationResult 摊平成若干行 dict。"""
    name = Path(item.image_path).name
    task = config.task_type
    data = item.result or {}

    if task == "box":
        for obj in data.get("objects", []):
            yield {
                "image": name,
                "label": obj.get("label", ""),
                "x1": _fmt(obj.get("x_min")),
                "y1": _fmt(obj.get("y_min")),
                "x2": _fmt(obj.get("x_max")),
                "y2": _fmt(obj.get("y_max")),
                "confidence": _fmt(obj.get("confidence")),
            }

    elif task == "point":
        for point in data.get("points", []):
            yield {
                "image": name,
                "label": point.get("label", ""),
                "x": _fmt(point.get("x")),
                "y": _fmt(point.get("y")),
                "confidence": _fmt(point.get("confidence")),
            }

    elif task == "ocr":
        for block in data.get("blocks", []):
            yield {
                "image": name,
                "text": block.get("text", ""),
                "type": block.get("type", ""),
                "x1": _fmt(block.get("x_min")),
                "y1": _fmt(block.get("y_min")),
                "x2": _fmt(block.get("x_max")),
                "y2": _fmt(block.get("y_max")),
            }

    elif task == "caption":
        yield {"image": name, "caption": data.get("summary", "")}

    elif task == "tag":
        for tag in data.get("tags", []):
            yield {
                "image": name,
                "label": tag.get("label", ""),
                "confidence": _fmt(tag.get("confidence")),
            }


def export_csv(results, config: Config) -> str:
    """写出/合并 annotations.csv，返回文件路径。"""
    os.makedirs(config.output_dir, exist_ok=True)
    csv_path = Path(config.output_dir) / CSV_NAME
    columns = COLUMNS[config.task_type]

    # 断点续跑：保留未参与本次重跑的历史行
    current_names = {Path(item.image_path).name for item in results}
    old_rows = []
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if (row.get("image") or "") not in current_names:
                        old_rows.append(row)
        except Exception:  # noqa: BLE001 - 文件损坏则重建
            old_rows = []

    new_rows = []
    for item in results:
        new_rows.extend(_rows_of(item, config))

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in old_rows:
            writer.writerow({col: row.get(col, "") for col in columns})
        for row in new_rows:
            writer.writerow({col: row.get(col, "") for col in columns})

    if config.resume:
        save_processed(
            config.output_dir,
            [relative_name(item.image_path, config.image_dir) for item in results],
        )

    return str(csv_path)


__all__ = ["export_csv", "COLUMNS"]

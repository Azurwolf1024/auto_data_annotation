# -*- coding: utf-8 -*-
"""YOLO 导出：只支持 box 任务。

输出：
- output_dir/labels/<stem>.txt：每行 `class_index x_center y_center width height`（全归一化）
- output_dir/classes.txt：类别名，行号即 class_index
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from ..config import Config


def _fmt(value) -> str:
    return f"{float(value):.6f}"


def to_yolo_line(obj: dict, targets: List[str]) -> str:
    """把一个检测框转成 YOLO 行。"""
    label = obj.get("label", "")
    try:
        class_index = targets.index(label)
    except ValueError:
        # 清洗阶段已保证 label 在 targets 中，这里做兜底防御
        raise ValueError(f"标签 {label!r} 不在 targets 列表中，无法生成 YOLO 标签")

    x_min = float(obj.get("x_min", 0.0))
    y_min = float(obj.get("y_min", 0.0))
    x_max = float(obj.get("x_max", 0.0))
    y_max = float(obj.get("y_max", 0.0))

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)

    return (
        f"{class_index} {_fmt(x_center)} {_fmt(y_center)} "
        f"{_fmt(width)} {_fmt(height)}"
    )


def export_yolo(results, config: Config) -> Tuple[str, str]:
    """写出 YOLO 标签，返回 (labels 目录, classes.txt 路径)。"""
    if config.task_type != "box":
        raise ValueError(
            f"yolo 格式只支持 box 任务，当前 task_type={config.task_type!r}"
        )

    labels_dir = Path(config.output_dir) / "labels"
    os.makedirs(labels_dir, exist_ok=True)

    classes_file = Path(config.output_dir) / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as handle:
        for name in config.targets:
            handle.write(f"{name}\n")

    for item in results:
        stem = Path(item.image_path).stem
        lines = [
            to_yolo_line(obj, config.targets)
            for obj in (item.result or {}).get("objects", [])
        ]
        out_path = labels_dir / f"{stem}.txt"
        with open(out_path, "w", encoding="utf-8") as handle:
            if lines:
                handle.write("\n".join(lines) + "\n")

    return str(labels_dir), str(classes_file)


__all__ = ["export_yolo", "to_yolo_line"]

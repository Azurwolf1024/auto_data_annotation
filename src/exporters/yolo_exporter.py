# -*- coding: utf-8 -*-
"""YOLO 导出：支持 box 与 point 两种任务。

输出：
- output_dir/labels/<stem>.txt
  - box   （标准检测格式）：`class_index x_center y_center width height`
  - point （YOLO-pose 格式）：`class_index x_center y_center width height kp_x kp_y kp_flag`
    其中 x_center/y_center 与 kp_x/kp_y 都取物体中心点，width/height 记 0
    （point 只标中心点，没有框），kp_flag 固定 2（可见且已标注）
- output_dir/classes.txt：类别名，行号即 class_index

所有数值均为归一化坐标（0~1），6 位小数。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from ..config import YOLO_TASK_TYPES, Config

# YOLO keypoint 可见性标志：0 未标注 / 1 标注但不可见 / 2 标注且可见
KP_VISIBLE = 2


def _fmt(value) -> str:
    return f"{float(value):.6f}"


def _class_index(label: str, targets: List[str]) -> int:
    try:
        return targets.index(label)
    except ValueError:
        # 清洗阶段已保证 label 在 targets 中，这里做兜底防御
        raise ValueError(f"标签 {label!r} 不在 targets 列表中，无法生成 YOLO 标签")


def to_yolo_line(obj: dict, targets: List[str]) -> str:
    """把一个检测框转成 YOLO 检测格式行（box 任务）。"""
    class_index = _class_index(obj.get("label", ""), targets)

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


def to_yolo_pose_line(point: dict, targets: List[str]) -> str:
    """把一个物体中心点转成 YOLO-pose 格式行（point 任务）。

    格式：class cx cy w h kp_x kp_y kp_flag
    point 只标中心点，因此 cx/cy 与 kp 相同，w/h 为 0，可见性固定 2。
    """
    class_index = _class_index(point.get("label", ""), targets)

    x = float(point.get("x", 0.0))
    y = float(point.get("y", 0.0))

    return (
        f"{class_index} {_fmt(x)} {_fmt(y)} {_fmt(0.0)} {_fmt(0.0)} "
        f"{_fmt(x)} {_fmt(y)} {KP_VISIBLE}"
    )


def _lines_of(item, config: Config) -> List[str]:
    """按任务类型把单张图的标注结果转成 YOLO 行列表。"""
    data = item.result or {}
    if config.task_type == "box":
        return [
            to_yolo_line(obj, config.targets) for obj in data.get("objects", [])
        ]
    if config.task_type == "point":
        return [
            to_yolo_pose_line(point, config.targets)
            for point in data.get("points", [])
        ]
    raise ValueError(
        f"yolo 格式只支持 {YOLO_TASK_TYPES} 任务，当前 task_type={config.task_type!r}"
    )


def export_yolo(results, config: Config) -> Tuple[str, str]:
    """写出 YOLO 标签，返回 (labels 目录, classes.txt 路径)。"""
    if config.task_type not in YOLO_TASK_TYPES:
        raise ValueError(
            f"yolo 格式只支持 {YOLO_TASK_TYPES} 任务，当前 task_type={config.task_type!r}"
        )

    labels_dir = Path(config.output_dir) / "labels"
    os.makedirs(labels_dir, exist_ok=True)

    classes_file = Path(config.output_dir) / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as handle:
        for name in config.targets:
            handle.write(f"{name}\n")

    for item in results:
        stem = Path(item.image_path).stem
        lines = _lines_of(item, config)
        out_path = labels_dir / f"{stem}.txt"
        with open(out_path, "w", encoding="utf-8") as handle:
            if lines:
                handle.write("\n".join(lines) + "\n")

    return str(labels_dir), str(classes_file)


__all__ = ["export_yolo", "to_yolo_line", "to_yolo_pose_line", "KP_VISIBLE"]

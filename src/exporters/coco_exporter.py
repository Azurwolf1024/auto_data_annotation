# -*- coding: utf-8 -*-
"""COCO 导出：支持 box（bbox）与 point（keypoints），整批汇总成 annotations.json。

坐标需要从归一化的 0~1 换算回绝对像素，因此需要读取图片真实宽高。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import read_image_size, relative_name, save_processed
from ..config import COCO_TASK_TYPES, Config

COCO_NAME = "annotations.json"


def _image_size(path: str) -> tuple:
    """读取图片宽高，失败时返回 (0, 0)（坐标退化为 0，但不中断整批）。"""
    try:
        width, height = read_image_size(path)
        return int(width), int(height)
    except Exception:  # noqa: BLE001
        return 0, 0


def _build_images(results, config: Config):
    """构建 COCO images 数组，并为每张图记录宽高。"""
    images = []
    sizes = {}
    for image_id, item in enumerate(results, start=1):
        width, height = _image_size(item.image_path)
        sizes[item.image_path] = (width, height)
        images.append(
            {
                "id": image_id,
                "file_name": Path(item.image_path).name,
                "width": width,
                "height": height,
            }
        )
    return images, sizes


def _build_categories(config: Config):
    """categories 来自 targets，id 从 1 开始。"""
    return [
        {"id": idx, "name": name, "supercategory": "object"}
        for idx, name in enumerate(config.targets, start=1)
    ]


def _build_annotations(results, config: Config, sizes, image_id_of, start_id: int = 1):
    """构建 COCO annotations 数组。"""
    annotations = []
    ann_id = start_id
    category_of = {name: idx for idx, name in enumerate(config.targets, start=1)}

    for item in results:
        width, height = sizes.get(item.image_path, (0, 0))
        image_id = image_id_of.get(item.image_path)
        if image_id is None:
            continue
        data = item.result or {}

        if config.task_type == "box":
            for obj in data.get("objects", []):
                category_id = category_of.get(obj.get("label"))
                if category_id is None:
                    continue
                x = float(obj.get("x_min", 0.0)) * width
                y = float(obj.get("y_min", 0.0)) * height
                w = (float(obj.get("x_max", 0.0)) - float(obj.get("x_min", 0.0))) * width
                h = (float(obj.get("y_max", 0.0)) - float(obj.get("y_min", 0.0))) * height
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [
                            round(x, 2),
                            round(y, 2),
                            round(max(w, 0.0), 2),
                            round(max(h, 0.0), 2),
                        ],
                        "area": round(max(w, 0.0) * max(h, 0.0), 2),
                        "iscrowd": 0,
                        "score": float(obj.get("confidence", 0.0)),
                    }
                )
                ann_id += 1

        elif config.task_type == "point":
            for point in data.get("points", []):
                category_id = category_of.get(point.get("label"))
                if category_id is None:
                    continue
                x = float(point.get("x", 0.0)) * width
                y = float(point.get("y", 0.0)) * height
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "keypoints": [round(x, 2), round(y, 2), 2],
                        "num_keypoints": 1,
                        "bbox": [round(x, 2), round(y, 2), 0.0, 0.0],
                        "area": 0.0,
                        "iscrowd": 0,
                        "score": float(point.get("confidence", 0.0)),
                    }
                )
                ann_id += 1

    return annotations


def export_coco(results, config: Config) -> str:
    """写出/合并 annotations.json，返回文件路径。"""
    if config.task_type not in COCO_TASK_TYPES:
        raise ValueError(
            f"coco 格式只支持 {COCO_TASK_TYPES} 任务，当前 task_type={config.task_type!r}"
        )

    os.makedirs(config.output_dir, exist_ok=True)
    out_path = Path(config.output_dir) / COCO_NAME

    current_names = {Path(item.image_path).name for item in results}

    old_images = []
    old_annotations = []
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as handle:
                old = json.load(handle)
            old_images = [
                img
                for img in old.get("images", [])
                if img.get("file_name") not in current_names
            ]
            keep_ids = {img["id"] for img in old_images}
            old_annotations = [
                ann
                for ann in old.get("annotations", [])
                if ann.get("image_id") in keep_ids
            ]
        except Exception:  # noqa: BLE001 - 文件损坏则重建
            old_images, old_annotations = [], []

    images, sizes = _build_images(results, config)
    image_id_of = {item.image_path: img["id"] for item, img in zip(results, images)}

    next_image_id = max([img["id"] for img in old_images], default=0) + 1
    for img in images:
        img["id"] = next_image_id
        next_image_id += 1
    image_id_of = {item.image_path: img["id"] for item, img in zip(results, images)}

    next_ann_id = max([ann["id"] for ann in old_annotations], default=0) + 1
    annotations = _build_annotations(results, config, sizes, image_id_of, next_ann_id)

    payload = {
        "info": {
            "description": "auto-annotator generated COCO dataset",
            "task_type": config.task_type,
            "model": config.model_name,
        },
        "licenses": [],
        "images": old_images + images,
        "annotations": old_annotations + annotations,
        "categories": _build_categories(config),
    }

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    if config.resume:
        save_processed(
            config.output_dir,
            [relative_name(item.image_path, config.image_dir) for item in results],
        )

    return str(out_path)


__all__ = ["export_coco"]

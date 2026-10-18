# -*- coding: utf-8 -*-
"""JSON 导出：每张图片一个 <stem>.json，放在 output_dir 下。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from ..config import Config


def export_json(results, config: Config) -> List[str]:
    """为每张成功处理的图片写出 output_dir/<stem>.json，返回写出文件列表。"""
    os.makedirs(config.output_dir, exist_ok=True)
    written: List[str] = []

    for item in results:
        stem = Path(item.image_path).stem
        out_path = Path(config.output_dir) / f"{stem}.json"
        payload = {
            "task_type": item.task_type,
            "image_path": item.image_path,
            "image": Path(item.image_path).name,
            "targets": list(config.targets),
            "model": config.model_name,
            "result": item.result,
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        written.append(str(out_path))

    return written


__all__ = ["export_json"]

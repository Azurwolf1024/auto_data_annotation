# -*- coding: utf-8 -*-
"""流水线模块：扫描图片 -> 调模型 -> 清洗 -> 封装结果。

支持：
- 断点续跑（json / yolo 检查同名文件；csv / coco 读 processed.json）
- 串行或 ThreadPoolExecutor 并发（--workers）
- 单张失败不影响整批，失败项封装 status="failed"
"""

from __future__ import annotations

import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Config
from .exporters import load_processed, read_image_size, relative_name
from .ollama_client import annotate
from .prompts import build_prompt
from .schema import SCHEMA_MAP, AnnotationResult
from .validator import clean

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# 每种任务的结果计数单位，用于进度日志
_COUNT_KEY = {
    "box": "objects",
    "point": "points",
    "ocr": "blocks",
    "tag": "tags",
}
_COUNT_UNIT = {
    "box": "objects",
    "point": "points",
    "ocr": "blocks",
    "tag": "tags",
    "caption": "caption",
}


# --------------------------------------------------------------------------- #
# 图片扫描
# --------------------------------------------------------------------------- #
def find_images(image_dir: str) -> List[Path]:
    """扫描目录下 jpg/jpeg/png/webp 图片（递归子目录，按路径排序）。"""
    root = Path(image_dir)
    images: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images, key=lambda p: str(p).lower())


# --------------------------------------------------------------------------- #
# 断点续跑
# --------------------------------------------------------------------------- #
def _single_file_output(config: Config, image: Path) -> Optional[Path]:
    """json / yolo 的单图输出文件路径；其他格式返回 None。"""
    stem = image.stem
    if config.output_format == "json":
        return Path(config.output_dir) / f"{stem}.json"
    if config.output_format == "yolo":
        return Path(config.output_dir) / "labels" / f"{stem}.txt"
    return None


def _is_done(config: Config, image: Path, processed: set) -> bool:
    """判断某张图是否已经处理过（断点续跑）。"""
    single = _single_file_output(config, image)
    if single is not None:
        return single.exists()
    return relative_name(str(image), config.image_dir) in processed


# --------------------------------------------------------------------------- #
# 单图处理
# --------------------------------------------------------------------------- #
def _count_of(task_type: str, result: Dict) -> int:
    if task_type == "caption":
        return 1 if result.get("summary") else 0
    key = _COUNT_KEY.get(task_type)
    value = result.get(key) if key else None
    return len(value) if isinstance(value, list) else 0


def annotate_image(image: Path, config: Config) -> AnnotationResult:
    """处理单张图片：build_prompt -> annotate -> clean -> 封装。"""
    task_type = config.task_type
    prompt = build_prompt(task_type, config.targets)
    schema_class = SCHEMA_MAP[task_type]

    raw = annotate(
        image_path=str(image),
        prompt=prompt,
        schema_class=schema_class,
        config=config,
    )

    image_size: Optional[Tuple[int, int]] = None
    if task_type in ("box", "point", "ocr"):
        try:
            image_size = read_image_size(str(image))
        except Exception:  # noqa: BLE001 - 读不到尺寸就跳过像素归一化
            image_size = None

    cleaned = clean(
        raw=raw,
        task_type=task_type,
        targets=config.targets,
        confidence_threshold=config.confidence_threshold,
        image_size=image_size,
    )
    return AnnotationResult(
        task_type=task_type,
        image_path=str(image),
        result=cleaned,
        status="ok",
        error=None,
    )


def _process_one(index: int, total: int, image: Path, config: Config) -> AnnotationResult:
    """单图处理 + 进度日志 + 异常兜底。"""
    try:
        result = annotate_image(image, config)
        count = _count_of(config.task_type, result.result)
        unit = _COUNT_UNIT.get(config.task_type, "items")
        print(f"[{index}/{total}] {image.name} → {count} {unit}", flush=True)
        return result
    except Exception as exc:  # noqa: BLE001 - 任何异常都要转成 failed 结果
        message = f"{type(exc).__name__}: {exc}"
        print(f"[{index}/{total}] {image.name} → 失败：{message}", flush=True)
        return AnnotationResult(
            task_type=config.task_type,
            image_path=str(image),
            result={},
            status="failed",
            error=message,
        )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(config: Config) -> List[AnnotationResult]:
    """按配置跑完整个目录，返回每张图的 AnnotationResult 列表。"""
    os.makedirs(config.output_dir, exist_ok=True)

    images = find_images(config.image_dir)
    total = len(images)
    if total == 0:
        print(f"未在 {config.image_dir} 下找到图片（支持 {sorted(IMAGE_EXTENSIONS)}）")
        return []

    print(f"配置：{config.describe()}")
    print(f"发现 {total} 张图片，开始处理…")

    processed = load_processed(config.output_dir) if config.resume else set()

    pending: List[Path] = []
    skipped = 0
    for image in images:
        if config.resume and _is_done(config, image, processed):
            skipped += 1
            continue
        pending.append(image)

    if skipped:
        print(f"断点续跑：跳过已处理的 {skipped} 张，待处理 {len(pending)} 张")

    results: List[AnnotationResult] = []
    if not pending:
        print("没有需要处理的图片。")
        return results

    tasks = [(idx, image) for idx, image in enumerate(pending, start=1)]
    total_pending = len(pending)

    if config.workers > 1:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            futures = [
                executor.submit(_process_one, idx, total_pending, image, config)
                for idx, image in tasks
            ]
            for future in futures:
                results.append(future.result())
    else:
        for idx, image in tasks:
            results.append(_process_one(idx, total_pending, image, config))

    failed = sum(1 for r in results if r.status != "ok")
    print(f"处理完成：成功 {len(results) - failed} 张，失败 {failed} 张")
    return results


__all__ = ["run", "annotate_image", "find_images"]

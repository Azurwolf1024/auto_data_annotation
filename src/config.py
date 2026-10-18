# -*- coding: utf-8 -*-
"""配置模块：Config 数据类与校验逻辑。"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List

# 本地 `ollama list` 确认的支持视觉（vision）的模型 tag
DEFAULT_MODEL = "qwen3.5:4b"

TASK_TYPES = ("box", "point", "ocr", "caption", "tag")
OUTPUT_FORMATS = ("json", "csv", "yolo", "coco")

# 各输出格式支持的 task_type
YOLO_TASK_TYPES = ("box", "point")  # point 用 YOLO-pose 行格式
COCO_TASK_TYPES = ("box", "point")


def parse_targets(raw) -> List[str]:
    """把逗号分隔的字符串解析成标签列表，去空白、去重、保序。"""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = str(raw).replace("，", ",").split(",")

    out: List[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


@dataclass
class Config:
    """全局运行配置。"""

    task_type: str
    targets: List[str] = field(default_factory=list)
    image_dir: str = ""
    output_dir: str = ""
    output_format: str = "json"
    model_name: str = DEFAULT_MODEL
    ollama_url: str = "http://localhost:11434"
    workers: int = 1
    confidence_threshold: float = 0.3
    resume: bool = True

    # ------------------------------------------------------------------ #
    def validate(self) -> "Config":
        """校验配置合法性，任何不合法都抛 ValueError。"""
        if self.task_type not in TASK_TYPES:
            raise ValueError(
                f"task_type 必须是 {TASK_TYPES} 之一，当前为 {self.task_type!r}"
            )

        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(
                f"output_format 必须是 {OUTPUT_FORMATS} 之一，当前为 {self.output_format!r}"
            )

        # 格式与任务类型的兼容性
        if self.output_format == "yolo" and self.task_type not in YOLO_TASK_TYPES:
            raise ValueError(
                f"yolo 格式只支持 {YOLO_TASK_TYPES} 任务，当前 task_type={self.task_type!r}"
            )

        if self.output_format == "coco" and self.task_type not in COCO_TASK_TYPES:
            raise ValueError(
                f"coco 格式只支持 {COCO_TASK_TYPES} 任务，当前 task_type={self.task_type!r}"
            )

        # box / point 是 closed-set，必须有候选标签
        if self.task_type in ("box", "point") and not self.targets:
            raise ValueError(
                f"task_type={self.task_type!r} 为闭集标注，必须通过 --targets 提供候选标签"
            )

        if not self.image_dir:
            raise ValueError("image_dir 不能为空")

        if not os.path.isdir(self.image_dir):
            raise ValueError(f"image_dir 不存在或不是目录：{self.image_dir}")

        if not self.output_dir:
            raise ValueError("output_dir 不能为空")

        if self.workers < 1:
            raise ValueError(f"workers 必须 >= 1，当前为 {self.workers}")

        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold 必须在 0~1 之间，当前为 {self.confidence_threshold}"
            )

        # 规范化
        self.targets = parse_targets(self.targets)
        self.task_type = self.task_type.strip().lower()
        self.output_format = self.output_format.strip().lower()
        self.image_dir = os.path.abspath(self.image_dir)
        self.output_dir = os.path.abspath(self.output_dir)
        return self

    # ------------------------------------------------------------------ #
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """从 argparse.Namespace 构造 Config。"""
        conf = getattr(args, "confidence", None)
        return cls(
            task_type=str(getattr(args, "task_type", "") or "").strip().lower(),
            targets=parse_targets(getattr(args, "targets", None)),
            image_dir=getattr(args, "image_dir", "") or "",
            output_dir=getattr(args, "output_dir", "") or "",
            output_format=str(getattr(args, "output_format", "json") or "json")
            .strip()
            .lower(),
            model_name=getattr(args, "model", None) or DEFAULT_MODEL,
            ollama_url=getattr(args, "ollama_url", None) or "http://localhost:11434",
            workers=int(getattr(args, "workers", 1) or 1),
            confidence_threshold=float(conf) if conf is not None else 0.3,
            resume=not bool(getattr(args, "no_resume", False)),
        )

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """一行人类可读的配置摘要，便于日志排查。"""
        return (
            f"task_type={self.task_type} format={self.output_format} "
            f"model={self.model_name} targets={self.targets or '-'} "
            f"workers={self.workers} conf>={self.confidence_threshold} "
            f"resume={self.resume}"
        )


__all__ = [
    "Config",
    "DEFAULT_MODEL",
    "TASK_TYPES",
    "OUTPUT_FORMATS",
    "YOLO_TASK_TYPES",
    "COCO_TASK_TYPES",
    "parse_targets",
]

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

# point 导出成框时的默认边长（像素）。零面积框没法训练，故默认给一个小方框
DEFAULT_POINT_BOX_SIZE = 32.0

# 二次验证（harness loop）默认参数
DEFAULT_VERIFY_ROUNDS = 1
DEFAULT_VERIFY_MARGIN = 0.25
DEFAULT_VERIFY_THRESHOLD = 0.5

# 放大精修（zoom-on-uncertain）默认参数
DEFAULT_ZOOM_UPSCALE = 2
DEFAULT_ZOOM_THRESHOLD = 0.3
DEFAULT_ZOOM_MAX_LEVELS = 2
DEFAULT_ZOOM_MARGIN = 0.25
DEFAULT_ZOOM_MODE = "model"  # model=模型主动请求 / threshold=低置信触发 / both=两者


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
    # point 导出 yolo/coco 时，把中心点膨胀成边长为该像素数的小正方形框；0 表示零面积
    point_box_size: float = DEFAULT_POINT_BOX_SIZE
    # 二次验证（harness loop）：检测后再逐候选裁切复核，剔除误检
    verify: bool = False
    verify_rounds: int = DEFAULT_VERIFY_ROUNDS
    verify_margin: float = DEFAULT_VERIFY_MARGIN
    verify_threshold: float = DEFAULT_VERIFY_THRESHOLD
    verify_domain_hints: str = ""
    # 放大精修（zoom-on-uncertain）：低置信度候选放大后重新标注并映射回原图
    zoom_on_uncertain: bool = False
    zoom_upscale: int = DEFAULT_ZOOM_UPSCALE
    zoom_threshold: float = DEFAULT_ZOOM_THRESHOLD
    zoom_max_levels: int = DEFAULT_ZOOM_MAX_LEVELS
    zoom_margin: float = DEFAULT_ZOOM_MARGIN
    zoom_via_mcp: bool = True
    zoom_server: str = ""
    # 放大触发方式：model=仅模型主动请求的 zoom_regions；threshold=仅低置信触发；
    # both=两者都触发。默认 model（模型主动 call 裁剪放大工具）。
    zoom_mode: str = DEFAULT_ZOOM_MODE
    # 自动可视化：把最终标注（框/点/文字框）画回原图供人工检查
    visualize: bool = False
    viz_dir: str = ""  # 可视化输出目录，默认 <output_dir>/_viz

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

        try:
            self.point_box_size = float(self.point_box_size)
        except (TypeError, ValueError):
            raise ValueError(f"point_box_size 必须是数字，当前为 {self.point_box_size!r}")
        if self.point_box_size < 0:
            raise ValueError(f"point_box_size 不能为负，当前为 {self.point_box_size}")

        # 二次验证（harness loop）仅支持 box / point
        if self.verify and self.task_type not in ("box", "point"):
            raise ValueError(
                f"--verify 仅支持 box / point 任务，当前 task_type={self.task_type!r}"
            )

        if self.verify:
            try:
                self.verify_rounds = int(self.verify_rounds)
            except (TypeError, ValueError):
                raise ValueError(f"verify_rounds 必须是正整数，当前为 {self.verify_rounds!r}")
            if self.verify_rounds < 1:
                raise ValueError(f"verify_rounds 必须 >= 1，当前为 {self.verify_rounds}")
            try:
                self.verify_margin = float(self.verify_margin)
            except (TypeError, ValueError):
                raise ValueError(f"verify_margin 必须是数字，当前为 {self.verify_margin!r}")
            if self.verify_margin < 0:
                raise ValueError(f"verify_margin 不能为负，当前为 {self.verify_margin}")
            try:
                self.verify_threshold = float(self.verify_threshold)
            except (TypeError, ValueError):
                raise ValueError(
                    f"verify_threshold 必须是数字，当前为 {self.verify_threshold!r}"
                )
            if not (0.0 <= self.verify_threshold <= 1.0):
                raise ValueError(
                    f"verify_threshold 必须在 0~1 之间，当前为 {self.verify_threshold}"
                )

        # 放大精修（zoom-on-uncertain）仅支持 box / point
        if self.zoom_on_uncertain and self.task_type not in ("box", "point"):
            raise ValueError(
                f"--zoom-on-uncertain 仅支持 box / point 任务，当前 task_type={self.task_type!r}"
            )

        if self.zoom_on_uncertain:
            if self.zoom_mode not in ("model", "threshold", "both"):
                raise ValueError(
                    f"zoom_mode 必须是 model/threshold/both 之一，当前为 {self.zoom_mode!r}"
                )
            try:
                self.zoom_upscale = int(self.zoom_upscale)
            except (TypeError, ValueError):
                raise ValueError(f"zoom_upscale 必须是整数，当前为 {self.zoom_upscale!r}")
            if self.zoom_upscale < 1:
                raise ValueError(f"zoom_upscale 必须 >= 1，当前为 {self.zoom_upscale}")

            try:
                self.zoom_max_levels = int(self.zoom_max_levels)
            except (TypeError, ValueError):
                raise ValueError(f"zoom_max_levels 必须是整数，当前为 {self.zoom_max_levels!r}")
            if self.zoom_max_levels < 1:
                raise ValueError(f"zoom_max_levels 必须 >= 1，当前为 {self.zoom_max_levels}")

            try:
                self.zoom_margin = float(self.zoom_margin)
            except (TypeError, ValueError):
                raise ValueError(f"zoom_margin 必须是数字，当前为 {self.zoom_margin!r}")
            if self.zoom_margin < 0:
                raise ValueError(f"zoom_margin 不能为负，当前为 {self.zoom_margin}")

            try:
                self.zoom_threshold = float(self.zoom_threshold)
            except (TypeError, ValueError):
                raise ValueError(
                    f"zoom_threshold 必须是数字，当前为 {self.zoom_threshold!r}"
                )
            if not (0.0 <= self.zoom_threshold <= 1.0):
                raise ValueError(
                    f"zoom_threshold 必须在 0~1 之间，当前为 {self.zoom_threshold}"
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
            point_box_size=(
                getattr(args, "point_box_size", None)
                if getattr(args, "point_box_size", None) is not None
                else DEFAULT_POINT_BOX_SIZE
            ),
            verify=bool(getattr(args, "verify", False)),
            verify_rounds=(
                int(getattr(args, "verify_rounds", DEFAULT_VERIFY_ROUNDS) or DEFAULT_VERIFY_ROUNDS)
            ),
            verify_margin=(
                float(getattr(args, "verify_margin", DEFAULT_VERIFY_MARGIN))
                if getattr(args, "verify_margin", None) is not None
                else DEFAULT_VERIFY_MARGIN
            ),
            verify_threshold=(
                float(getattr(args, "verify_threshold", DEFAULT_VERIFY_THRESHOLD))
                if getattr(args, "verify_threshold", None) is not None
                else DEFAULT_VERIFY_THRESHOLD
            ),
            verify_domain_hints=str(getattr(args, "verify_domain_hints", "") or ""),
            zoom_on_uncertain=bool(getattr(args, "zoom_on_uncertain", False)),
            zoom_upscale=(
                int(getattr(args, "zoom_upscale", DEFAULT_ZOOM_UPSCALE) or DEFAULT_ZOOM_UPSCALE)
            ),
            zoom_threshold=(
                float(getattr(args, "zoom_threshold", DEFAULT_ZOOM_THRESHOLD))
                if getattr(args, "zoom_threshold", None) is not None
                else DEFAULT_ZOOM_THRESHOLD
            ),
            zoom_max_levels=(
                int(getattr(args, "zoom_max_levels", DEFAULT_ZOOM_MAX_LEVELS) or DEFAULT_ZOOM_MAX_LEVELS)
            ),
            zoom_margin=(
                float(getattr(args, "zoom_margin", DEFAULT_ZOOM_MARGIN))
                if getattr(args, "zoom_margin", None) is not None
                else DEFAULT_ZOOM_MARGIN
            ),
            zoom_via_mcp=bool(getattr(args, "zoom_via_mcp", True)),
            zoom_server=str(getattr(args, "zoom_server", "") or ""),
            zoom_mode=str(
                getattr(args, "zoom_mode", DEFAULT_ZOOM_MODE) or DEFAULT_ZOOM_MODE
            )
            .strip()
            .lower(),
            visualize=bool(getattr(args, "visualize", False)),
            viz_dir=str(getattr(args, "viz_dir", "") or ""),
        )

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """一行人类可读的配置摘要，便于日志排查。"""
        return (
            f"task_type={self.task_type} format={self.output_format} "
            f"model={self.model_name} targets={self.targets or '-'} "
            f"workers={self.workers} conf>={self.confidence_threshold} "
            f"resume={self.resume}"
            + (
                f" verify=on rounds={self.verify_rounds} margin={self.verify_margin} "
                f"thr={self.verify_threshold}"
                if self.verify
                else ""
            )
            + (
                f" zoom=on up={self.zoom_upscale} thr={self.zoom_threshold} "
                f"levels={self.zoom_max_levels} mode={self.zoom_mode} "
                f"mcp={self.zoom_via_mcp}"
                if self.zoom_on_uncertain
                else ""
            )
            + (f" visualize=on dir={self.viz_dir or '<output>/_viz'}" if self.visualize else "")
        )


__all__ = [
    "Config",
    "DEFAULT_MODEL",
    "TASK_TYPES",
    "OUTPUT_FORMATS",
    "YOLO_TASK_TYPES",
    "COCO_TASK_TYPES",
    "DEFAULT_POINT_BOX_SIZE",
    "DEFAULT_VERIFY_ROUNDS",
    "DEFAULT_VERIFY_MARGIN",
    "DEFAULT_VERIFY_THRESHOLD",
    "DEFAULT_ZOOM_UPSCALE",
    "DEFAULT_ZOOM_THRESHOLD",
    "DEFAULT_ZOOM_MAX_LEVELS",
    "DEFAULT_ZOOM_MARGIN",
    "DEFAULT_ZOOM_MODE",
    "parse_targets",
]

# -*- coding: utf-8 -*-
"""导出器包：根据 output_format 分发到具体导出实现。

同时提供两个跨模块共享的小工具：
- read_image_size：纯 Python 读取图片宽高（jpg/jpeg/png/webp），避免额外依赖；
- load_processed / save_processed：csv / coco 的断点续跑状态文件。
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROCESSED_FILENAME = "processed.json"


# --------------------------------------------------------------------------- #
# 图片尺寸（纯 Python，无第三方依赖）
# --------------------------------------------------------------------------- #
def _png_size(data: bytes) -> Tuple[int, int]:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    raise ValueError("不是合法的 PNG 数据")


def _jpeg_size(data: bytes) -> Tuple[int, int]:
    if not data or data[0] != 0xFF or data[1] != 0xD8:
        raise ValueError("不是合法的 JPEG 数据")
    pos = 2
    length = len(data)
    while pos < length - 1:
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker in (0xD8, 0x01, 0xD9, 0xFF):
            pos += 2
            continue
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height, width = struct.unpack(">HH", data[pos + 5 : pos + 9])
            return int(width), int(height)
        if pos + 4 > length:
            break
        segment = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        pos += 2 + int(segment)
    raise ValueError("JPEG 中未找到 SOF 标记")


def _webp_size(data: bytes) -> Tuple[int, int]:
    if len(data) < 16 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("不是合法的 WebP 数据")
    fmt = data[12:16]
    if fmt == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if fmt == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    if fmt == b"VP8 ":
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height
    raise ValueError(f"不支持的 WebP 子格式：{fmt!r}")


def read_image_size(path: str) -> Tuple[int, int]:
    """读取图片宽高，返回 (width, height)。

    优先用纯 Python 解析文件头；解析失败时若安装了 Pillow 则回退到 Pillow。
    """
    with open(path, "rb") as handle:
        head = handle.read(64)
        with open(path, "rb") as full:
            full.seek(0)
            data = full.read() if len(head) >= 16 else b""

    suffix = os.path.splitext(str(path))[1].lower()
    parsers = []
    if suffix in (".png",):
        parsers = [_png_size]
    elif suffix in (".jpg", ".jpeg"):
        parsers = [_jpeg_size]
    elif suffix == ".webp":
        parsers = [_webp_size]
    else:
        parsers = [_png_size, _jpeg_size, _webp_size]

    for parser in parsers:
        try:
            width, height = parser(data)
            if width > 0 and height > 0:
                return width, height
        except Exception:  # noqa: BLE001 - 换下一个解析器
            continue

    try:  # 最后尝试 Pillow
        from PIL import Image  # type: ignore

        with Image.open(str(path)) as img:
            return img.size
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法读取图片尺寸：{path}（{exc}）") from exc


# --------------------------------------------------------------------------- #
# point -> 小方框（零面积框没法训练，需要把中心点膨胀成正方形框）
# --------------------------------------------------------------------------- #
DEFAULT_POINT_FALLBACK_NORM = 0.02  # 取不到图片尺寸时，按图片短边的 2% 兜底


def point_box_norm(
    x: float,
    y: float,
    width: int,
    height: int,
    box_size_px: float,
) -> Tuple[float, float, float, float]:
    """把物体中心点膨胀成一个小正方形框。

    参数
    ----
    x, y : 归一化中心点坐标（0~1）
    width, height : 图片真实像素宽高（未知传 0）
    box_size_px : 方框边长（像素）；<=0 表示不膨胀，返回零面积框

    返回
    ----
    (cx, cy, w_norm, h_norm)：框中心与归一化宽高，框会向内收敛以保证完整落在图内
    """
    x = min(max(float(x), 0.0), 1.0)
    y = min(max(float(y), 0.0), 1.0)
    box_size_px = float(box_size_px or 0.0)

    if box_size_px <= 0.0:
        return x, y, 0.0, 0.0

    if width > 0 and height > 0:
        w_norm = box_size_px / float(width)
        h_norm = box_size_px / float(height)
    else:
        # 取不到图片尺寸时的兜底：按归一化边长理解（<1 视为归一化，否则取默认比例）
        w_norm = h_norm = (
            box_size_px if 0.0 < box_size_px < 1.0 else DEFAULT_POINT_FALLBACK_NORM
        )

    w_norm = min(w_norm, 1.0)
    h_norm = min(h_norm, 1.0)

    # 让方框完整落在图片内（必要时微调中心，关键点坐标仍保持模型原值）
    cx = min(max(x, w_norm / 2.0), 1.0 - w_norm / 2.0)
    cy = min(max(y, h_norm / 2.0), 1.0 - h_norm / 2.0)
    return cx, cy, w_norm, h_norm


# --------------------------------------------------------------------------- #
# 断点续跑状态
# --------------------------------------------------------------------------- #
def load_processed(output_dir: str) -> set:
    """读取 processed.json，返回已处理图片相对路径集合。"""
    path = Path(output_dir) / PROCESSED_FILENAME
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            names = data.get("processed", [])
        elif isinstance(data, list):
            names = data
        else:
            names = []
        return set(str(n) for n in names)
    except Exception:  # noqa: BLE001 - 状态文件损坏时按空处理
        return set()


def save_processed(output_dir: str, names: Iterable[str]) -> None:
    """合并写回 processed.json。"""
    path = Path(output_dir) / PROCESSED_FILENAME
    existing = load_processed(output_dir)
    for name in names:
        existing.add(str(name))
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {"processed": sorted(existing)}, handle, ensure_ascii=False, indent=2
        )


def relative_name(image_path: str, image_dir: str) -> str:
    """图片相对 image_dir 的路径（统一正斜杠），用于断点记录与跨模块对齐。"""
    try:
        return (
            Path(image_path).resolve().relative_to(Path(image_dir).resolve()).as_posix()
        )
    except (ValueError, OSError):
        return Path(image_path).name


# --------------------------------------------------------------------------- #
# 分发
# --------------------------------------------------------------------------- #
from . import coco_exporter, csv_exporter, json_exporter, yolo_exporter  # noqa: E402
from .coco_exporter import export_coco  # noqa: E402
from .csv_exporter import export_csv  # noqa: E402
from .json_exporter import export_json  # noqa: E402
from .yolo_exporter import export_yolo  # noqa: E402


def _ok_results(results) -> List:
    """只保留处理成功的结果。"""
    return [r for r in results if getattr(r, "status", "ok") == "ok"]


def export(results, config) -> Dict[str, str]:
    """根据 config.output_format 分发导出，返回输出文件说明。"""
    ok = _ok_results(results)
    fmt = config.output_format

    if fmt == "json":
        files = export_json(ok, config)
        return {"format": "json", "path": str(config.output_dir), "files": str(len(files))}

    if fmt == "csv":
        path = export_csv(ok, config)
        return {"format": "csv", "path": str(path), "files": "1"}

    if fmt == "yolo":
        labels_dir, classes_file = export_yolo(ok, config)
        return {
            "format": "yolo",
            "path": str(labels_dir),
            "files": str(len(ok)),
            "classes": str(classes_file),
        }

    if fmt == "coco":
        path = export_coco(ok, config)
        return {"format": "coco", "path": str(path), "files": "1"}

    raise ValueError(f"未知 output_format：{fmt!r}")


__all__ = [
    "export",
    "export_json",
    "export_csv",
    "export_yolo",
    "export_coco",
    "read_image_size",
    "point_box_norm",
    "relative_name",
    "load_processed",
    "save_processed",
    "PROCESSED_FILENAME",
]

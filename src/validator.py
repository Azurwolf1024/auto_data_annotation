# -*- coding: utf-8 -*-
"""校验/清洗模块：把模型原始输出整理成规范结果。

处理规则：
- box / point：丢弃不在 targets 中的标签、丢弃低置信度、裁剪坐标到 [0,1]、
  坐标顺序错误时交换；
- tag：同 box 的过滤逻辑，另外超过 10 个截断；
- ocr：只裁剪坐标，不做标签与置信度过滤；
- caption：原样返回。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

MAX_TAGS = 10

_COORD_FIELDS_BOX = ("x_min", "y_min", "x_max", "y_max")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _to_float(value: Any, default: float = 0.0) -> float:
    """安全转 float，失败返回默认值。"""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float) -> float:
    """裁剪到 [0, 1]。"""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _normalize_coord(value: float, size: Optional[int]) -> float:
    """如果提供了图片宽/高且坐标明显是像素值（>1），则归一化。"""
    if size and size > 0 and value > 1.0:
        return value / float(size)
    return value


def _fix_pair(lo: float, hi: float) -> Tuple[float, float]:
    """保证 lo <= hi（顺序错误则交换）。"""
    return (lo, hi) if lo <= hi else (hi, lo)


def _match_target(label: Any, targets: List[str]) -> Optional[str]:
    """把模型输出的标签映射到 candidates 中的标准写法。

    先精确匹配，再退化为去空白 + 忽略大小写匹配；都不中返回 None。
    """
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    for target in targets:
        if text == target:
            return target
    lowered = text.lower()
    for target in targets:
        if str(target).lower() == lowered:
            return target
    return None


def _ensure_list(raw: Any, key: str) -> List[Dict[str, Any]]:
    """取出 raw 中的列表字段，兼容模型少给/多给一层嵌套的情况。"""
    value = raw.get(key) if isinstance(raw, dict) else None
    if value is None and isinstance(raw, dict):
        # 部分模型会把结构包在 "result" / "data" 之类的键里
        for alt in ("result", "data", "items"):
            nested = raw.get(alt)
            if isinstance(nested, dict) and isinstance(nested.get(key), list):
                value = nested[key]
                break
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


# --------------------------------------------------------------------------- #
# 各任务清洗
# --------------------------------------------------------------------------- #
def _clean_box_items(
    raw: Dict[str, Any],
    targets: List[str],
    threshold: float,
    image_size: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    width = image_size[0] if image_size else None
    height = image_size[1] if image_size else None

    cleaned: List[Dict[str, Any]] = []
    for item in _ensure_list(raw, "objects"):
        label = _match_target(item.get("label"), targets)
        if label is None:
            continue
        confidence = _to_float(item.get("confidence"), 0.0)
        confidence = clamp(confidence)
        if confidence < threshold:
            continue

        x_min = _normalize_coord(_to_float(item.get("x_min")), width)
        y_min = _normalize_coord(_to_float(item.get("y_min")), height)
        x_max = _normalize_coord(_to_float(item.get("x_max")), width)
        y_max = _normalize_coord(_to_float(item.get("y_max")), height)

        x_min, x_max = _fix_pair(x_min, x_max)
        y_min, y_max = _fix_pair(y_min, y_max)

        x_min, y_min, x_max, y_max = (
            clamp(x_min),
            clamp(y_min),
            clamp(x_max),
            clamp(y_max),
        )
        if x_max - x_min <= 0.0 or y_max - y_min <= 0.0:
            continue  # 退化框直接丢弃

        cleaned.append(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "x_min": round(x_min, 6),
                "y_min": round(y_min, 6),
                "x_max": round(x_max, 6),
                "y_max": round(y_max, 6),
            }
        )
    return cleaned


def _clean_point_items(
    raw: Dict[str, Any],
    targets: List[str],
    threshold: float,
    image_size: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    width = image_size[0] if image_size else None
    height = image_size[1] if image_size else None

    cleaned: List[Dict[str, Any]] = []
    for item in _ensure_list(raw, "points"):
        label = _match_target(item.get("label"), targets)
        if label is None:
            continue
        confidence = clamp(_to_float(item.get("confidence"), 0.0))
        if confidence < threshold:
            continue

        x = clamp(_normalize_coord(_to_float(item.get("x")), width))
        y = clamp(_normalize_coord(_to_float(item.get("y")), height))

        cleaned.append(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "x": round(x, 6),
                "y": round(y, 6),
            }
        )
    return cleaned


def _clean_ocr_items(
    raw: Dict[str, Any],
    image_size: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    width = image_size[0] if image_size else None
    height = image_size[1] if image_size else None

    cleaned: List[Dict[str, Any]] = []
    for item in _ensure_list(raw, "blocks"):
        text = item.get("text")
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue

        x_min = _normalize_coord(_to_float(item.get("x_min")), width)
        y_min = _normalize_coord(_to_float(item.get("y_min")), height)
        x_max = _normalize_coord(_to_float(item.get("x_max")), width)
        y_max = _normalize_coord(_to_float(item.get("y_max")), height)

        x_min, x_max = _fix_pair(x_min, x_max)
        y_min, y_max = _fix_pair(y_min, y_max)

        block_type = item.get("type")
        block_type = str(block_type).strip() if block_type else "other"

        cleaned.append(
            {
                "text": text,
                "type": block_type,
                "x_min": round(clamp(x_min), 6),
                "y_min": round(clamp(y_min), 6),
                "x_max": round(clamp(x_max), 6),
                "y_max": round(clamp(y_max), 6),
            }
        )
    return cleaned


def _clean_tag_items(
    raw: Dict[str, Any],
    targets: List[str],
    threshold: float,
) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for item in _ensure_list(raw, "tags"):
        raw_label = item.get("label")
        if raw_label is None:
            continue
        label = str(raw_label).strip()
        if not label:
            continue

        # tag 是候选池：给了 targets 时优先映射到池中标签，
        # 未命中则允许保留模型自创标签
        if targets:
            label = _match_target(label, targets) or label

        confidence = clamp(_to_float(item.get("confidence"), 1.0))
        if confidence < threshold:
            continue

        cleaned.append({"label": label, "confidence": round(confidence, 4)})

    cleaned.sort(key=lambda t: t["confidence"], reverse=True)
    return cleaned[:MAX_TAGS]


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def clean(
    raw: Dict[str, Any],
    task_type: str,
    targets: List[str] = None,
    confidence_threshold: float = 0.3,
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """清洗模型原始输出。

    参数
    ----
    raw : 模型返回的 dict
    task_type : box / point / ocr / caption / tag
    targets : 候选标签列表
    confidence_threshold : 置信度阈值，低于该值的项被丢弃
    image_size : 可选 (width, height)，用于把模型误输出的像素坐标归一化

    返回
    ----
    清洗后的 dict，结构与 schema.py 中对应模型一致
    """
    if not isinstance(raw, dict):
        return {}

    targets = list(targets or [])
    task_type = (task_type or "").strip().lower()

    if task_type == "box":
        return {"objects": _clean_box_items(raw, targets, confidence_threshold, image_size)}

    if task_type == "point":
        return {"points": _clean_point_items(raw, targets, confidence_threshold, image_size)}

    if task_type == "ocr":
        return {"blocks": _clean_ocr_items(raw, image_size)}

    if task_type == "caption":
        summary = raw.get("summary", "")
        if not isinstance(summary, str):
            summary = str(summary)
        return {"summary": summary.strip()}

    if task_type == "tag":
        return {"tags": _clean_tag_items(raw, targets, confidence_threshold)}

    raise ValueError(f"未知 task_type：{task_type!r}")


__all__ = ["clean", "clamp", "MAX_TAGS"]

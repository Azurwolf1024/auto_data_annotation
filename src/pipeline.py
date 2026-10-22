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
from .mcp_zoom import make_zoom_client
from .ollama_client import annotate, build_client
from .prompts import build_prompt, build_verify_prompt
from .schema import (
    SCHEMA_MAP,
    AnnotationResult,
    BoxAnnotation,
    PointAnnotation,
    VerifyAnnotation,
)
from .validator import clean, _match_target, clamp

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
def find_images(image_dir: str, exclude_dir: Optional[str] = None) -> List[Path]:
    """扫描目录下 jpg/jpeg/png/webp 图片（递归子目录，按路径排序）。

    exclude_dir 用于跳过输出目录子树（例如 --verify 会在 output_dir/_verify/
    下生成裁剪 PNG，若 output_dir 恰好位于 image_dir 内部，需避免把它们当成
    待标注图片重复处理）。
    """
    root = Path(image_dir)
    exclude = Path(exclude_dir).resolve() if exclude_dir else None
    images: List[Path] = []
    for path in root.rglob("*"):
        if not (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
            continue
        if exclude is not None and exclude in [p.resolve() for p in path.parents]:
            continue
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
def _to_float(value, default: float = 0.0) -> float:
    """安全转 float，失败返回默认值。"""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# 二次验证（harness loop）
# --------------------------------------------------------------------------- #
def _crop_norm_around_box(
    box: Tuple[float, float, float, float],
    margin: float,
    image_size: Tuple[int, int],
) -> Tuple[Tuple[float, float, float, float], Tuple[int, int, int, int]]:
    """以候选框为中心、按 margin 比例向外扩边，返回裁切区（归一化 + 像素）。

    margin 解释为「相对框尺寸的单侧扩边比例」：左右各扩 margin*宽、上下各扩 margin*高。
    像素坐标做了边界 clamp，保证裁切区域非空且落在图内。
    """
    x_min, y_min, x_max, y_max = (clamp(v) for v in box)
    cw = max(x_max - x_min, 1e-6)
    ch = max(y_max - y_min, 1e-6)

    nx0 = clamp(x_min - margin * cw)
    nx1 = clamp(x_max + margin * cw)
    ny0 = clamp(y_min - margin * ch)
    ny1 = clamp(y_max + margin * ch)

    width, height = image_size
    px0 = max(0, min(int(round(nx0 * width)), width))
    px1 = max(0, min(int(round(nx1 * width)), width))
    py0 = max(0, min(int(round(ny0 * height)), height))
    py1 = max(0, min(int(round(ny1 * height)), height))
    if px1 <= px0:
        px1 = min(px0 + 1, width)
    if py1 <= py0:
        py1 = min(py0 + 1, height)

    return (nx0, ny0, nx1, ny1), (px0, py0, px1, py1)


def _verify_boxes(
    image_path: str,
    objects: List[Dict[str, Any]],
    config: Config,
    pil_image,
    image_size: Tuple[int, int],
    client,
) -> List[Dict[str, Any]]:
    """对 box 任务的每个候选框逐轮裁切复核，返回通过验证的框。"""
    from PIL import Image

    targets = config.targets
    stem = Path(image_path).stem
    verify_dir = Path(config.output_dir) / "_verify" / stem
    os.makedirs(verify_dir, exist_ok=True)

    verified: List[Dict[str, Any]] = []
    for idx, obj in enumerate(objects, start=1):
        box = (obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"])
        accepted = False
        last_box = box
        last_label = ""
        last_conf = 0.0

        for rnd in range(1, config.verify_rounds + 1):
            crop_norm, (px0, py0, px1, py1) = _crop_norm_around_box(
                box, config.verify_margin, image_size
            )
            crop = pil_image.crop((px0, py0, px1, py1))
            crop_path = verify_dir / f"cand{idx:02d}_r{rnd}.png"
            crop.save(str(crop_path))

            prompt = build_verify_prompt(targets, config.verify_domain_hints)
            try:
                raw = annotate(
                    str(crop_path), prompt, VerifyAnnotation, config, client=client
                )
            except Exception as exc:  # noqa: BLE001 - 复核调用失败则剔除该候选
                print(f"    ✗ cand{idx} 第{rnd}轮复核调用失败：{exc}", flush=True)
                accepted = False
                break

            present = bool(raw.get("present"))
            label = _match_target(raw.get("label"), targets) or ""
            conf = clamp(_to_float(raw.get("confidence"), 0.0))
            if not present or not label or conf < config.verify_threshold:
                print(
                    f"    ✗ cand{idx} 第{rnd}轮被拒 present={present} "
                    f"label={raw.get('label')!r} conf={conf:.2f}",
                    flush=True,
                )
                accepted = False
                break

            # 把裁切内的修正框反投影回全图坐标
            cx0, cy0, cx1, cy1 = crop_norm
            cw = cx1 - cx0
            ch = cy1 - cy0
            bx0 = clamp(cx0 + _to_float(raw.get("x_min"), 0.0) * cw)
            by0 = clamp(cy0 + _to_float(raw.get("y_min"), 0.0) * ch)
            bx1 = clamp(cx0 + _to_float(raw.get("x_max"), 0.0) * cw)
            by1 = clamp(cy0 + _to_float(raw.get("y_max"), 0.0) * ch)
            bx0, bx1 = (bx0, bx1) if bx0 <= bx1 else (bx1, bx0)
            by0, by1 = (by0, by1) if by0 <= by1 else (by1, by0)
            box = (bx0, by0, bx1, by1)
            last_box = box
            last_label = label
            last_conf = conf
            accepted = True
            print(
                f"    ✓ cand{idx} 第{rnd}轮接受 label={label} conf={conf:.2f}",
                flush=True,
            )

        if accepted:
            bx0, by0, bx1, by1 = last_box
            verified.append(
                {
                    "label": last_label,
                    "confidence": round(last_conf, 4),
                    "x_min": round(bx0, 6),
                    "y_min": round(by0, 6),
                    "x_max": round(bx1, 6),
                    "y_max": round(by1, 6),
                }
            )

    print(
        f"  二次验证：保留 {len(verified)}/{len(objects)} 个框",
        flush=True,
    )
    return verified


def _verify_points(
    image_path: str,
    points: List[Dict[str, Any]],
    config: Config,
    pil_image,
    image_size: Tuple[int, int],
    client,
) -> List[Dict[str, Any]]:
    """对 point 任务的每个候选点逐轮裁切复核，返回通过验证的点。

    point 没有框尺寸，裁切半边长直接用 verify_margin（归一化）；接受时取
    模型给出的修正框中心点作为最终点坐标（更准）。
    """
    targets = config.targets
    width, height = image_size
    stem = Path(image_path).stem
    verify_dir = Path(config.output_dir) / "_verify" / stem
    os.makedirs(verify_dir, exist_ok=True)

    verified: List[Dict[str, Any]] = []
    for idx, pt in enumerate(points, start=1):
        x, y = clamp(_to_float(pt.get("x"))), clamp(_to_float(pt.get("y")))
        accepted = False
        cur_x, cur_y = x, y
        last_label = ""
        last_conf = 0.0

        for rnd in range(1, config.verify_rounds + 1):
            nx0 = clamp(x - config.verify_margin)
            nx1 = clamp(x + config.verify_margin)
            ny0 = clamp(y - config.verify_margin)
            ny1 = clamp(y + config.verify_margin)
            px0 = max(0, min(int(round(nx0 * width)), width))
            px1 = max(0, min(int(round(nx1 * width)), width))
            py0 = max(0, min(int(round(ny0 * height)), height))
            py1 = max(0, min(int(round(ny1 * height)), height))
            if px1 <= px0:
                px1 = min(px0 + 1, width)
            if py1 <= py0:
                py1 = min(py0 + 1, height)

            crop = pil_image.crop((px0, py0, px1, py1))
            crop_path = verify_dir / f"cand{idx:02d}_r{rnd}.png"
            crop.save(str(crop_path))

            prompt = build_verify_prompt(targets, config.verify_domain_hints)
            try:
                raw = annotate(
                    str(crop_path), prompt, VerifyAnnotation, config, client=client
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    ✗ cand{idx} 第{rnd}轮复核调用失败：{exc}", flush=True)
                accepted = False
                break

            present = bool(raw.get("present"))
            label = _match_target(raw.get("label"), targets) or ""
            conf = clamp(_to_float(raw.get("confidence"), 0.0))
            if not present or not label or conf < config.verify_threshold:
                print(
                    f"    ✗ cand{idx} 第{rnd}轮被拒 present={present} "
                    f"label={raw.get('label')!r} conf={conf:.2f}",
                    flush=True,
                )
                accepted = False
                break

            # 反投影修正框中心作为新点坐标
            cw = nx1 - nx0
            ch = ny1 - ny0
            bx0 = nx0 + _to_float(raw.get("x_min"), 0.0) * cw
            by0 = ny0 + _to_float(raw.get("y_min"), 0.0) * ch
            bx1 = nx0 + _to_float(raw.get("x_max"), 0.0) * cw
            by1 = ny0 + _to_float(raw.get("y_max"), 0.0) * ch
            cur_x = clamp((bx0 + bx1) / 2.0)
            cur_y = clamp((by0 + by1) / 2.0)
            x, y = cur_x, cur_y
            last_label = label
            last_conf = conf
            accepted = True
            print(
                f"    ✓ cand{idx} 第{rnd}轮接受 label={label} conf={conf:.2f}",
                flush=True,
            )

        if accepted:
            verified.append(
                {
                    "label": last_label,
                    "confidence": round(last_conf, 4),
                    "x": round(cur_x, 6),
                    "y": round(cur_y, 6),
                }
            )

    print(
        f"  二次验证：保留 {len(verified)}/{len(points)} 个点",
        flush=True,
    )
    return verified


def harness_verify(
    image_path: str,
    cleaned: Dict[str, Any],
    config: Config,
    pil_image,
    image_size: Tuple[int, int],
    client=None,
) -> Dict[str, Any]:
    """检测之后的二次验证入口，按任务类型分派到 box / point 复核。"""
    if config.task_type == "box":
        verified = _verify_boxes(
            image_path, cleaned.get("objects", []), config, pil_image, image_size, client
        )
        return {"objects": verified}
    if config.task_type == "point":
        verified = _verify_points(
            image_path, cleaned.get("points", []), config, pil_image, image_size, client
        )
        return {"points": verified}
    return cleaned


# --------------------------------------------------------------------------- #
# 放大精修（zoom-on-uncertain）：模型不确定时放大、重新标注、映射回原图
# --------------------------------------------------------------------------- #
def _norm_region(r) -> Optional[Tuple[float, float, float, float]]:
    """把 dict / 4 元组归一化区域解析成 (x0,y0,x1,y1)，非法或退化返回 None。"""
    try:
        if isinstance(r, dict):
            x0, y0, x1, y1 = r["x_min"], r["y_min"], r["x_max"], r["y_max"]
        elif isinstance(r, (tuple, list)) and len(r) == 4:
            x0, y0, x1, y1 = r
        else:
            return None
        x0, y0, x1, y1 = clamp(float(x0)), clamp(float(y0)), clamp(float(x1)), clamp(float(y1))
    except (KeyError, TypeError, ValueError):
        return None
    if x1 - x0 <= 0.0 or y1 - y0 <= 0.0:
        return None
    return (x0, y0, x1, y1)


def _normalize_regions(regions, image_size=None) -> List[Dict[str, Any]]:
    """把模型返回的 zoom_regions 整理成合法归一化区域。

    小模型经常把坐标写成**像素值**而不是 0~1（与它输出 box 坐标时同样的毛病）。
    这类区域若直接丢弃，「模型主动放大」会静默失效，因此这里按图片尺寸换算回来，
    再交给 _norm_region 做 clamp 与合法性过滤。

    返回形如 [{"x_min","y_min","x_max","y_max","reason"}, ...] 的列表。
    """
    width = height = None
    if image_size:
        try:
            width, height = int(image_size[0]), int(image_size[1])
        except (TypeError, ValueError, IndexError):
            width = height = None

    out: List[Dict[str, Any]] = []
    for r in regions or []:
        try:
            if isinstance(r, dict):
                x0 = float(r["x_min"]); y0 = float(r["y_min"])
                x1 = float(r["x_max"]); y1 = float(r["y_max"])
                reason = str(r.get("reason", "") or "")
            elif isinstance(r, (list, tuple)) and len(r) == 4:
                x0, y0, x1, y1 = (float(v) for v in r)
                reason = ""
            else:
                continue
        except (KeyError, TypeError, ValueError):
            continue

        # 明显超出 0~1 → 判定为像素坐标，按图片尺寸归一化
        if max(x0, x1, y0, y1) > 1.0 and width and height:
            x0, x1 = x0 / width, x1 / width
            y0, y1 = y0 / height, y1 / height

        reg = _norm_region((x0, y0, x1, y1))
        if reg:
            out.append(
                {
                    "x_min": reg[0], "y_min": reg[1],
                    "x_max": reg[2], "y_max": reg[3],
                    "reason": reason,
                }
            )
    return out


def _region_of_obj(obj: Dict[str, Any], task_type: str, margin: float):
    """从已有检测项构造裁切区域：box 用其框；point 用点周围方形（半边长 margin）。"""
    if task_type == "box":
        return _norm_region((obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]))
    x = clamp(obj.get("x", 0.0))
    y = clamp(obj.get("y", 0.0))
    m = max(0.0, float(margin))
    return _norm_region((clamp(x - m), clamp(y - m), clamp(x + m), clamp(y + m)))


def _iou(box_a, box_b) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    a = (ax1 - ax0) * (ay1 - ay0)
    b = (bx1 - bx0) * (by1 - by0)
    union = a + b - inter
    return inter / union if union > 0 else 0.0


def _collect_seeds(task_type: str, cleaned: Dict[str, Any], model_regions, config):
    """收集需要放大的种子区域。

    - model / both 模式：模型在结构化输出里声明的 zoom_regions（模型主动 call 工具）；
    - threshold / both 模式：低置信候选（旧行为）。
    """
    seeds: List[Dict[str, Any]] = []
    if config.zoom_mode in ("model", "both"):
        for r in model_regions or []:
            reg = _norm_region(r)
            if reg:
                seeds.append({"region": reg, "label": None, "src": "model"})
    if config.zoom_mode in ("threshold", "both"):
        key = "objects" if task_type == "box" else "points"
        for o in cleaned.get(key, []):
            if float(o.get("confidence", 0.0)) < config.zoom_threshold:
                reg = _region_of_obj(o, task_type, config.zoom_margin)
                if reg:
                    seeds.append({"region": reg, "label": o.get("label"), "src": "threshold"})
    return seeds


def _zoom_chain(
    image_path,
    full_region,
    seed_region,
    task_type,
    config,
    zoom_client,
    zoom_dir,
    depth,
    label_filter,
    tag_prefix,
):
    """对一个种子区域执行一次放大+重标注，并递归处理裁切图里的 zoom_regions。

    返回：相对【原图】坐标系的检测项列表（dict）。
    """
    is_box = task_type == "box"
    tag = f"{tag_prefix}_L{depth}"
    try:
        data = zoom_client.crop_and_upscale(
            image_path, full_region, seed_region, config.zoom_upscale, str(zoom_dir), tag=tag
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    ✗ 放大调用失败（{tag}）：{exc}", flush=True)
        return []
    crop_path = data["cropped_image_path"]
    cropped_region = data["region"]  # 该裁切图相对原图的矩形（已 compose）
    try:
        raw = annotate(
            crop_path, build_prompt(task_type, config.targets), SCHEMA_MAP[task_type], config
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    ✗ 放大标注失败（{tag}）：{exc}", flush=True)
        return []

    dets = clean(raw, task_type, config.targets, 0.0)
    key = "objects" if is_box else "points"
    items = dets.get(key, [])
    if label_filter:
        items = [d for d in items if d.get("label") == label_filter]

    mapped: List[Dict[str, Any]] = []
    for d in items:
        try:
            if is_box:
                o = zoom_client.map_box_to_original(
                    cropped_region, (d["x_min"], d["y_min"], d["x_max"], d["y_max"])
                )
                mapped.append(
                    {
                        "label": d["label"],
                        "confidence": round(float(d["confidence"]), 4),
                        "x_min": round(o["x_min"], 6),
                        "y_min": round(o["y_min"], 6),
                        "x_max": round(o["x_max"], 6),
                        "y_max": round(o["y_max"], 6),
                    }
                )
            else:
                o = zoom_client.map_point_to_original(cropped_region, d["x"], d["y"])
                mapped.append(
                    {
                        "label": d["label"],
                        "confidence": round(float(d["confidence"]), 4),
                        "x": round(o["x"], 6),
                        "y": round(o["y"], 6),
                    }
                )
        except Exception:  # noqa: BLE001
            continue

    # 模型驱动的多级放大：裁切图若再声明 zoom_regions，则继续更深一层
    # （裁切图里的坐标同样可能是像素值，按裁切图自身尺寸换算）
    if depth < config.zoom_max_levels and config.zoom_mode in ("model", "both"):
        sub_regions = _normalize_regions(raw.get("zoom_regions") or [], data.get("size"))
        for j, r in enumerate(sub_regions, start=1):
            sub = (r["x_min"], r["y_min"], r["x_max"], r["y_max"])
            mapped += _zoom_chain(
                crop_path,
                cropped_region,
                sub,
                task_type,
                config,
                zoom_client,
                zoom_dir,
                depth + 1,
                None,
                f"{tag_prefix}_s{j}",
            )
    return mapped


def _merge_detections(existing, additions, task_type, iou_thr=0.5) -> List[Dict[str, Any]]:
    """把放大得到的新检测合并进原结果：与同标签已有项 IoU/距离重合则替换，否则追加。"""
    merged = [dict(e) for e in existing]
    for add in additions:
        replaced = False
        if task_type == "box":
            for i, ex in enumerate(merged):
                if ex.get("label") == add.get("label") and _iou(
                    (ex["x_min"], ex["y_min"], ex["x_max"], ex["y_max"]),
                    (add["x_min"], add["y_min"], add["x_max"], add["y_max"]),
                ) >= iou_thr:
                    merged[i] = add
                    replaced = True
                    break
        else:
            for i, ex in enumerate(merged):
                d = ((ex["x"] - add["x"]) ** 2 + (ex["y"] - add["y"]) ** 2) ** 0.5
                if ex.get("label") == add.get("label") and d <= 0.02:
                    merged[i] = add
                    replaced = True
                    break
        if not replaced:
            merged.append(add)
    return merged


def zoom_refine(
    image_path: str,
    cleaned: Dict[str, Any],
    config: Config,
    zoom_client,
    model_regions=None,
) -> Dict[str, Any]:
    """放大精修入口（模型主动请求 +/或低置信触发），按任务类型处理并合并结果。

    模型在结构化输出里声明的 zoom_regions 会作为种子；若 zoom_mode 含
    threshold，低置信候选也会作为种子。每个种子调用裁剪放大工具、重新标注、
    映射回原图，最终按 IoU/距离合并（重合的同类检测以精修结果替换，新发现则追加）。
    """
    if config.task_type not in ("box", "point"):
        return cleaned

    stem = Path(image_path).stem
    zoom_dir = Path(config.output_dir) / "_zoom" / stem
    os.makedirs(zoom_dir, exist_ok=True)

    seeds = _collect_seeds(config.task_type, cleaned, model_regions, config)
    additions: List[Dict[str, Any]] = []
    for idx, seed in enumerate(seeds, start=1):
        additions += _zoom_chain(
            image_path,
            (0.0, 0.0, 1.0, 1.0),  # 原图相对自身的矩形
            seed["region"],
            config.task_type,
            config,
            zoom_client,
            zoom_dir,
            1,
            seed["label"],
            f"s{idx}",
        )

    if config.task_type == "box":
        cleaned["objects"] = _merge_detections(cleaned.get("objects", []), additions, "box")
    else:
        cleaned["points"] = _merge_detections(cleaned.get("points", []), additions, "point")

    print(
        f"  放大精修：{len(additions)} 个精修结果来自 {len(seeds)} 个种子区域",
        flush=True,
    )
    return cleaned


def _count_of(task_type: str, result: Dict) -> int:
    if task_type == "caption":
        return 1 if result.get("summary") else 0
    key = _COUNT_KEY.get(task_type)
    value = result.get(key) if key else None
    return len(value) if isinstance(value, list) else 0


def annotate_image(image: Path, config: Config, zoom_client=None) -> AnnotationResult:
    """处理单张图片：build_prompt -> annotate -> clean -> (可选)二次验证 -> 封装。

    zoom_client 可选：由 run() 在整批处理中复用的放大客户端（MCP / 本地）。
    若未传入且本图需要放大精修，则临时新建并在本函数内关闭（直接调用本函数的
    场景，如单测）。
    """
    task_type = config.task_type
    prompt = build_prompt(task_type, config.targets)
    schema_class = SCHEMA_MAP[task_type]

    # 开启二次验证或放大精修时，复用同一个 Ollama 客户端（检测 + 多轮调用）
    client = None
    if task_type in ("box", "point") and (config.verify or config.zoom_on_uncertain):
        try:
            client = build_client(getattr(config, "ollama_url", "http://localhost:11434"))
        except Exception:  # noqa: BLE001 - 建客户端失败则退回每次自建
            client = None

    raw = annotate(
        image_path=str(image),
        prompt=prompt,
        schema_class=schema_class,
        config=config,
        client=client,
    )

    image_size: Optional[Tuple[int, int]] = None
    pil_image = None
    if task_type in ("box", "point", "ocr"):
        try:
            from PIL import Image  # 延迟导入，避免非验证场景硬依赖 Pillow

            pil_image = Image.open(str(image))
            image_size = pil_image.size
        except Exception:  # noqa: BLE001 - 读不到尺寸就跳过像素归一化
            pil_image = None
            try:
                image_size = read_image_size(str(image))
            except Exception:
                image_size = None

    cleaned = clean(
        raw=raw,
        task_type=task_type,
        targets=config.targets,
        confidence_threshold=config.confidence_threshold,
        image_size=image_size,
    )

    # 模型可能主动声明 zoom_regions（模型主动 call 裁剪放大工具）。
    # 必须在拿到 image_size 之后再整理：模型可能给的是像素坐标，需要换算回 0~1。
    model_regions = _normalize_regions(raw.get("zoom_regions") or [], image_size)

    # 二次验证（harness loop）：box / point 检测后再逐候选裁切复核
    if config.verify and task_type in ("box", "point"):
        if pil_image is not None and image_size is not None:
            cleaned = harness_verify(
                image_path=str(image),
                cleaned=cleaned,
                config=config,
                pil_image=pil_image,
                image_size=image_size,
                client=client,
            )
        else:
            print(f"  ⚠ {image.name} 无法读取图片尺寸，跳过二次验证", flush=True)

    # 放大精修（zoom-on-uncertain）：低置信度候选放大后重新标注并映射回原图
    if config.zoom_on_uncertain and task_type in ("box", "point"):
        if pil_image is not None and image_size is not None:
            own_zoom = zoom_client is None
            try:
                if own_zoom:
                    zoom_client = make_zoom_client(config)
                cleaned = zoom_refine(
                    image_path=str(image),
                    cleaned=cleaned,
                    config=config,
                    zoom_client=zoom_client,
                    model_regions=model_regions,
                )
            except Exception as exc:  # noqa: BLE001 - 放大失败则保留原结果
                print(f"  ⚠ {image.name} 放大精修失败，保留原结果：{exc}", flush=True)
            finally:
                if own_zoom and zoom_client is not None:
                    zoom_client.close()
        else:
            print(f"  ⚠ {image.name} 无法读取图片尺寸，跳过放大精修", flush=True)

    # 自动可视化：把最终标注（框/点/文字框）画回原图，供人工检查
    if config.visualize and task_type in ("box", "point", "ocr"):
        try:
            from .visualize import draw_result
            viz_dir = config.viz_dir or os.path.join(config.output_dir, "_viz")
            os.makedirs(viz_dir, exist_ok=True)
            viz_path = os.path.join(viz_dir, f"{Path(image).stem}.png")
            draw_result(str(image), cleaned, config.task_type, viz_path)
            print(f"  可视化标注图：{viz_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ {image.name} 可视化失败：{exc}", flush=True)

    return AnnotationResult(
        task_type=task_type,
        image_path=str(image),
        result=cleaned,
        status="ok",
        error=None,
    )


def _process_one(index: int, total: int, image: Path, config: Config, zoom_client=None) -> AnnotationResult:
    """单图处理 + 进度日志 + 异常兜底。"""
    try:
        result = annotate_image(image, config, zoom_client=zoom_client)
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

    images = find_images(config.image_dir, exclude_dir=config.output_dir)
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

    # 放大精修需要的 MCP / 本地客户端在整批内只创建一次并复用：
    # - 跨图复用避免每张图都拉起/销毁子进程；
    # - MCPZoomClient 已加锁，--workers 并发下共享也安全。
    shared_zoom = None
    if config.zoom_on_uncertain and config.task_type in ("box", "point"):
        try:
            shared_zoom = make_zoom_client(config)
            print(f"  放大服务已就绪（{type(shared_zoom).__name__}）", flush=True)
        except Exception as exc:  # noqa: BLE001 - 启动失败则每张图内部回退
            print(f"  ⚠ 放大服务启动失败，改为每张图内回退：{exc}", flush=True)
            shared_zoom = None

    tasks = [(idx, image) for idx, image in enumerate(pending, start=1)]
    total_pending = len(pending)

    try:
        if config.workers > 1:
            with ThreadPoolExecutor(max_workers=config.workers) as executor:
                futures = [
                    executor.submit(
                        _process_one, idx, total_pending, image, config, shared_zoom
                    )
                    for idx, image in tasks
                ]
                for future in futures:
                    results.append(future.result())
        else:
            for idx, image in tasks:
                results.append(
                    _process_one(idx, total_pending, image, config, shared_zoom)
                )
    finally:
        if shared_zoom is not None:
            shared_zoom.close()

    failed = sum(1 for r in results if r.status != "ok")
    print(f"处理完成：成功 {len(results) - failed} 张，失败 {failed} 张")
    return results


__all__ = [
    "run",
    "annotate_image",
    "find_images",
    "harness_verify",
    "_verify_boxes",
    "_verify_points",
    "zoom_refine",
]

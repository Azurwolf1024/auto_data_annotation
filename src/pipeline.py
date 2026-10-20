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
def _zoom_refine_boxes(
    image_path: str,
    objects: List[Dict[str, Any]],
    config: Config,
    zoom_client,
) -> List[Dict[str, Any]]:
    """对置信度低于阈值的 box 候选放大精修，支持多级（递归）放大。

    流程（单候选）：
    1. 以候选框为 local 子矩形，调用裁切放大服务得到裁切图 + 原图矩形 region；
    2. 把裁切图再次喂给 box 检测模型，筛出同类目标；
    3. 若置信度仍低于阈值且未达最大级数，以该检测框为新的 local、以上一级
       region 为新 region，进入更深一层放大（坐标随 region 一路 compose 回原图）；
    4. 命中或耗尽级数后，用最终的 region 把局部框映射回原图坐标，替换原候选。
    """
    targets = config.targets
    stem = Path(image_path).stem
    zoom_dir = Path(config.output_dir) / "_zoom" / stem
    os.makedirs(zoom_dir, exist_ok=True)

    refined: List[Dict[str, Any]] = []
    for idx, obj in enumerate(objects, start=1):
        if float(obj.get("confidence", 0.0)) >= config.zoom_threshold:
            refined.append(obj)
            continue

        region = {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
        cur_image = image_path
        local = (obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"])
        best = None
        best_region = region

        for lvl in range(1, config.zoom_max_levels + 1):
            try:
                data = zoom_client.crop_and_upscale(
                    cur_image, region, local, config.zoom_upscale, str(zoom_dir),
                    tag=f"cand{idx}_L{lvl}",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    ✗ cand{idx} 第{lvl}级放大调用失败：{exc}", flush=True)
                break
            crop_path = data["cropped_image_path"]
            composed = data["region"]
            try:
                raw = annotate(
                    crop_path, build_prompt("box", targets), BoxAnnotation, config
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    ✗ cand{idx} 第{lvl}级放大标注失败：{exc}", flush=True)
                break

            dets = clean(raw, "box", targets, 0.0).get("objects", [])
            dets = [d for d in dets if d.get("label") == obj.get("label")]
            if not dets:
                print(f"    · cand{idx} 第{lvl}级放大未检出同类目标", flush=True)
                break

            cur = max(dets, key=lambda d: float(d.get("confidence", 0.0)))
            best = cur
            best_region = composed
            if float(cur.get("confidence", 0.0)) >= config.zoom_threshold:
                print(
                    f"    ✓ cand{idx} 第{lvl}级放大后置信度提升至 "
                    f"{cur['confidence']:.2f}",
                    flush=True,
                )
                break
            # 继续更深放大：以当前检测框为新的 local（相对本裁切图）
            local = (cur["x_min"], cur["y_min"], cur["x_max"], cur["y_max"])
            region = composed
            cur_image = crop_path

        if best is not None:
            orig = zoom_client.map_box_to_original(
                best_region, (best["x_min"], best["y_min"], best["x_max"], best["y_max"])
            )
            refined.append(
                {
                    "label": obj["label"],
                    "confidence": round(float(best.get("confidence", obj.get("confidence", 0.0))), 4),
                    "x_min": round(orig["x_min"], 6),
                    "y_min": round(orig["y_min"], 6),
                    "x_max": round(orig["x_max"], 6),
                    "y_max": round(orig["y_max"], 6),
                }
            )
        else:
            refined.append(obj)

    print(f"  放大精修：{len(refined)}/{len(objects)} 个框完成", flush=True)
    return refined


def _zoom_refine_points(
    image_path: str,
    points: List[Dict[str, Any]],
    config: Config,
    zoom_client,
) -> List[Dict[str, Any]]:
    """对置信度低于阈值的 point 候选放大精修（围绕点取方形区域裁切），支持多级。"""
    targets = config.targets
    stem = Path(image_path).stem
    zoom_dir = Path(config.output_dir) / "_zoom" / stem
    os.makedirs(zoom_dir, exist_ok=True)
    m = config.zoom_margin

    refined: List[Dict[str, Any]] = []
    for idx, pt in enumerate(points, start=1):
        if float(pt.get("confidence", 0.0)) >= config.zoom_threshold:
            refined.append(pt)
            continue

        x = clamp(pt.get("x", 0.0))
        y = clamp(pt.get("y", 0.0))
        region = {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
        cur_image = image_path
        local = (clamp(x - m), clamp(y - m), clamp(x + m), clamp(y + m))
        best = None
        best_region = region

        for lvl in range(1, config.zoom_max_levels + 1):
            try:
                data = zoom_client.crop_and_upscale(
                    cur_image, region, local, config.zoom_upscale, str(zoom_dir),
                    tag=f"pt{idx}_L{lvl}",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    ✗ pt{idx} 第{lvl}级放大调用失败：{exc}", flush=True)
                break
            crop_path = data["cropped_image_path"]
            composed = data["region"]
            try:
                raw = annotate(
                    crop_path, build_prompt("point", targets), PointAnnotation, config
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    ✗ pt{idx} 第{lvl}级放大标注失败：{exc}", flush=True)
                break

            dets = clean(raw, "point", targets, 0.0).get("points", [])
            dets = [d for d in dets if d.get("label") == pt.get("label")]
            if not dets:
                print(f"    · pt{idx} 第{lvl}级放大未检出同类目标", flush=True)
                break

            cur = max(dets, key=lambda d: float(d.get("confidence", 0.0)))
            best = cur
            best_region = composed
            if float(cur.get("confidence", 0.0)) >= config.zoom_threshold:
                print(
                    f"    ✓ pt{idx} 第{lvl}级放大后置信度提升至 "
                    f"{cur['confidence']:.2f}",
                    flush=True,
                )
                break
            cx = clamp(cur.get("x", 0.0))
            cy = clamp(cur.get("y", 0.0))
            local = (clamp(cx - m), clamp(cy - m), clamp(cx + m), clamp(cy + m))
            region = composed
            cur_image = crop_path

        if best is not None:
            orig = zoom_client.map_point_to_original(best_region, best.get("x", 0.0), best.get("y", 0.0))
            refined.append(
                {
                    "label": pt["label"],
                    "confidence": round(float(best.get("confidence", pt.get("confidence", 0.0))), 4),
                    "x": round(orig["x"], 6),
                    "y": round(orig["y"], 6),
                }
            )
        else:
            refined.append(pt)

    print(f"  放大精修：{len(refined)}/{len(points)} 个点完成", flush=True)
    return refined


def zoom_refine(
    image_path: str,
    cleaned: Dict[str, Any],
    config: Config,
    zoom_client,
) -> Dict[str, Any]:
    """放大精修入口，按任务类型分派到 box / point。"""
    if config.task_type == "box":
        cleaned["objects"] = _zoom_refine_boxes(
            image_path, cleaned.get("objects", []), config, zoom_client
        )
    elif config.task_type == "point":
        cleaned["points"] = _zoom_refine_points(
            image_path, cleaned.get("points", []), config, zoom_client
        )
    return cleaned


def _count_of(task_type: str, result: Dict) -> int:
    if task_type == "caption":
        return 1 if result.get("summary") else 0
    key = _COUNT_KEY.get(task_type)
    value = result.get(key) if key else None
    return len(value) if isinstance(value, list) else 0


def annotate_image(image: Path, config: Config) -> AnnotationResult:
    """处理单张图片：build_prompt -> annotate -> clean -> (可选)二次验证 -> 封装。"""
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
            try:
                zoom_client = make_zoom_client(config)
                cleaned = zoom_refine(
                    image_path=str(image),
                    cleaned=cleaned,
                    config=config,
                    zoom_client=zoom_client,
                )
                zoom_client.close()
            except Exception as exc:  # noqa: BLE001 - 放大失败则保留原结果
                print(f"  ⚠ {image.name} 放大精修失败，保留原结果：{exc}", flush=True)
        else:
            print(f"  ⚠ {image.name} 无法读取图片尺寸，跳过放大精修", flush=True)

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


__all__ = [
    "run",
    "annotate_image",
    "find_images",
    "harness_verify",
    "_verify_boxes",
    "_verify_points",
    "zoom_refine",
    "_zoom_refine_boxes",
    "_zoom_refine_points",
]

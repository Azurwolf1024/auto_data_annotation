# -*- coding: utf-8 -*-
"""harness 循环中「模型主动放大」与「自动可视化」的集成测试。

用桩模型（fake_annotate）确定性驱动，走**真实 MCP 子进程**，不依赖 Ollama 推理。

覆盖场景：
1. 模型主动声明 zoom_regions → harness 调裁剪放大工具 → 结果映射回原图（box 单级）；
2. 裁切图里再次声明 zoom_regions → 递归多级放大，坐标逐级 compose 回原图；
3. point 任务的模型驱动放大；
4. zoom_mode=threshold（旧行为：仅低置信触发）仍然工作；
5. 模型没声明 zoom_regions 时，model 模式不产生多余裁切；
6. harness verify 二次复核的反投影；
7. --visualize 生成带框可视化图（默认目录与自定义目录）；
8. pipeline.run 多图场景下放大客户端整批复用。

桩模型按**裁切层级**区分响应：裁切图文件名形如 ``crop_s1_L1.png``（tag 由
pipeline 传入），因此可从中解析出层级；原图没有 ``_L<n>`` 后缀，视为第 0 级。

运行：
    python tests/test_harness_zoom.py
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要 Pillow：pip install -r requirements.txt") from exc

import src.pipeline as pipeline  # noqa: E402
from src.config import Config  # noqa: E402


# --------------------------------------------------------------------------- #
# 桩模型
# --------------------------------------------------------------------------- #
# SCRIPT[level] = {"objects"/"points": [...], "zoom_regions": [...]}
# level 0 = 原图；level N = 第 N 级裁切图。SCRIPT["verify"] 给 verify 复核用。
SCRIPT: dict = {}


def _level_of(image_path: str) -> int:
    m = re.search(r"_L(\d+)\.png$", os.path.basename(image_path))
    return int(m.group(1)) if m else 0


def fake_annotate(image_path, prompt, schema_class, config, max_attempts=3, client=None):
    name = schema_class.__name__

    if name == "VerifyAnnotation":
        return SCRIPT.get("verify", {"present": True, "label": "red circle",
                                     "confidence": 0.9,
                                     "x_min": 0.1, "y_min": 0.1,
                                     "x_max": 0.9, "y_max": 0.9})

    entry = SCRIPT.get(_level_of(image_path), {})
    regions = entry.get("zoom_regions", [])
    if name == "BoxAnnotation":
        return {"objects": entry.get("objects", []), "zoom_regions": regions}
    if name == "PointAnnotation":
        return {"points": entry.get("points", []), "zoom_regions": regions}
    raise AssertionError(f"未预期的 schema：{name}")


def approx(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


def make_image(path, w=1000, h=800, color=(200, 80, 80)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (w, h), color).save(path)


def check(cond, msg):
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  ✓ {msg}")


def run_annotate(img_path, cfg):
    """用桩模型跑 annotate_image，跑完恢复真实 annotate。"""
    global SCRIPT
    orig = pipeline.annotate
    pipeline.annotate = fake_annotate
    try:
        return pipeline.annotate_image(img_path, cfg)
    finally:
        pipeline.annotate = orig


def find_box(objs, label):
    for o in objs:
        if o.get("label") == label:
            return o
    return None


# --------------------------------------------------------------------------- #
# 1. 模型主动放大：box 单级
# --------------------------------------------------------------------------- #
def test_model_driven_box_single_level(img_dir, out_dir):
    print("[test] 模型主动 zoom_regions → box 单级放大并映射回原图")
    global SCRIPT
    img = os.path.join(img_dir, "a.png")
    make_image(img)

    SCRIPT = {
        0: {
            "objects": [
                {"label": "red circle", "confidence": 0.85,
                 "x_min": 0.05, "y_min": 0.05, "x_max": 0.15, "y_max": 0.15}
            ],
            # 模型主动说：这块太小，我要放大看
            "zoom_regions": [
                {"x_min": 0.4, "y_min": 0.3, "x_max": 0.6, "y_max": 0.5, "reason": "目标太小"}
            ],
        },
        1: {
            "objects": [
                {"label": "blue square", "confidence": 0.92,
                 "x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}
            ],
            "zoom_regions": [],
        },
    }

    cfg = Config(
        task_type="box", targets=["red circle", "blue square"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2,
        zoom_on_uncertain=True, zoom_via_mcp=True, zoom_mode="model",
        zoom_max_levels=1, zoom_upscale=2,
    )

    res = run_annotate(img, cfg)
    objs = res.result.get("objects", [])
    check(len(objs) == 2, f"原检测 + 放大新发现共 2 个框（实际 {len(objs)}）")

    zoomed = find_box(objs, "blue square")
    check(zoomed is not None, "放大区域检出的 blue square 被合并进结果")
    if zoomed:
        # 裁切区 (0.4,0.3,0.6,0.5) 内 (0.1,0.1,0.9,0.9) → 原图 (0.42,0.32,0.58,0.48)
        check(
            approx(zoomed["x_min"], 0.42) and approx(zoomed["x_max"], 0.58)
            and approx(zoomed["y_min"], 0.32) and approx(zoomed["y_max"], 0.48),
            f"放大检测映射回原图 {zoomed} ≈ (0.42,0.32,0.58,0.48)",
        )
        check(zoomed["confidence"] == 0.92, f"置信度来自裁切图 {zoomed['confidence']}")

    # 确实生成了裁切产物（证明真的调了 MCP 裁剪放大工具）
    crops = list(_walk_png(os.path.join(out_dir, "_zoom")))
    check(len(crops) >= 1, f"生成了 {len(crops)} 个裁切放大产物（MCP 真实调用）")


# --------------------------------------------------------------------------- #
# 2. 模型主动放大：多级递归
# --------------------------------------------------------------------------- #
def test_model_driven_box_multilevel(img_dir, out_dir):
    print("[test] 模型主动 zoom_regions → box 多级递归放大（坐标逐级 compose）")
    global SCRIPT
    img = os.path.join(img_dir, "b.png")
    make_image(img)

    SCRIPT = {
        0: {
            "objects": [
                {"label": "red circle", "confidence": 0.85,
                 "x_min": 0.05, "y_min": 0.05, "x_max": 0.15, "y_max": 0.15}
            ],
            "zoom_regions": [{"x_min": 0.4, "y_min": 0.3, "x_max": 0.8, "y_max": 0.7}],
        },
        1: {
            "objects": [
                {"label": "blue square", "confidence": 0.9,
                 "x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4}
            ],
            # 第 1 级裁切图里模型又要求放大右下角
            "zoom_regions": [{"x_min": 0.6, "y_min": 0.6, "x_max": 1.0, "y_max": 1.0}],
        },
        2: {
            "objects": [
                {"label": "red circle", "confidence": 0.95,
                 "x_min": 0.2, "y_min": 0.2, "x_max": 0.6, "y_max": 0.6}
            ],
            "zoom_regions": [],
        },
    }

    cfg = Config(
        task_type="box", targets=["red circle", "blue square"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2,
        zoom_on_uncertain=True, zoom_via_mcp=True, zoom_mode="model",
        zoom_max_levels=2, zoom_upscale=2,
    )

    res = run_annotate(img, cfg)
    objs = res.result.get("objects", [])
    check(len(objs) == 3, f"原检测 + 两级放大共 3 个框（实际 {len(objs)}）")

    lvl1 = find_box(objs, "blue square")
    check(lvl1 is not None, "第 1 级放大的 blue square 存在")
    if lvl1:
        # (0.4,0.3,0.8,0.7) 内 (0.1,0.1,0.4,0.4) → (0.44,0.34,0.56,0.46)
        check(
            approx(lvl1["x_min"], 0.44) and approx(lvl1["x_max"], 0.56)
            and approx(lvl1["y_min"], 0.34) and approx(lvl1["y_max"], 0.46),
            f"第 1 级映射 {lvl1} ≈ (0.44,0.34,0.56,0.46)",
        )

    # 第 2 级：compose((0.4,0.3,0.8,0.7), (0.6,0.6,1.0,1.0)) = (0.64,0.54,0.8,0.7)
    # 内部 (0.2,0.2,0.6,0.6) → (0.672,0.572,0.736,0.636)
    lvl2 = [o for o in objs if o is not lvl1 and o.get("label") == "red circle"
            and o["x_min"] > 0.5]
    check(len(lvl2) == 1, "第 2 级放大的 red circle 存在")
    if lvl2:
        b = lvl2[0]
        check(
            approx(b["x_min"], 0.672, 1e-3) and approx(b["x_max"], 0.736, 1e-3)
            and approx(b["y_min"], 0.572, 1e-3) and approx(b["y_max"], 0.636, 1e-3),
            f"第 2 级多级 compose 映射 {b} ≈ (0.672,0.572,0.736,0.636)",
        )


# --------------------------------------------------------------------------- #
# 3. 模型主动放大：point
# --------------------------------------------------------------------------- #
def test_model_driven_point(img_dir, out_dir):
    print("[test] 模型主动 zoom_regions → point 放大并映射回原图")
    global SCRIPT
    img = os.path.join(img_dir, "c.png")
    make_image(img, color=(80, 160, 200))

    SCRIPT = {
        0: {
            "points": [{"label": "red circle", "confidence": 0.85, "x": 0.1, "y": 0.1}],
            "zoom_regions": [{"x_min": 0.4, "y_min": 0.3, "x_max": 0.8, "y_max": 0.7}],
        },
        1: {
            "points": [{"label": "red circle", "confidence": 0.95, "x": 0.5, "y": 0.5}],
            "zoom_regions": [],
        },
    }

    cfg = Config(
        task_type="point", targets=["red circle"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2,
        zoom_on_uncertain=True, zoom_via_mcp=True, zoom_mode="model",
        zoom_max_levels=1, zoom_upscale=2,
    )

    res = run_annotate(img, cfg)
    pts = res.result.get("points", [])
    check(len(pts) == 2, f"原点 + 放大新发现共 2 个点（实际 {len(pts)}）")
    zoomed = [p for p in pts if p["x"] > 0.3]
    check(len(zoomed) == 1, "放大得到的点存在")
    if zoomed:
        p = zoomed[0]
        # (0.4,0.3,0.8,0.7) 内 (0.5,0.5) → (0.6, 0.5)
        check(approx(p["x"], 0.6) and approx(p["y"], 0.5),
              f"point 映射回原图 {p} ≈ (0.6,0.5)")


# --------------------------------------------------------------------------- #
# 3.5 模型把 zoom_regions 写成像素坐标 → 自动换算回归一化后再放大
# --------------------------------------------------------------------------- #
def test_pixel_coord_zoom_regions(img_dir, out_dir):
    print("[test] zoom_regions 给的是像素坐标 → 按图片尺寸归一化后仍触发放大")
    global SCRIPT
    img = os.path.join(img_dir, "c2.png")
    make_image(img, color=(70, 170, 140))

    SCRIPT = {
        0: {
            "objects": [],
            # 1000x800 图上 (400,240)-(600,400) 像素 = 归一化 (0.4,0.3)-(0.6,0.5)
            "zoom_regions": [
                {"x_min": 400, "y_min": 240, "x_max": 600, "y_max": 400, "reason": "px"}
            ],
        },
        1: {
            "objects": [
                {"label": "red circle", "confidence": 0.9,
                 "x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}
            ],
            "zoom_regions": [],
        },
    }

    cfg = Config(
        task_type="box", targets=["red circle"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2,
        zoom_on_uncertain=True, zoom_via_mcp=True, zoom_mode="model",
        zoom_max_levels=1, zoom_upscale=2,
    )

    res = run_annotate(img, cfg)
    objs = res.result.get("objects", [])
    check(len(objs) == 1, f"像素坐标区域被换算后触发了放大（实际 {len(objs)} 个框）")
    if objs:
        b = objs[0]
        check(
            approx(b["x_min"], 0.42) and approx(b["x_max"], 0.58)
            and approx(b["y_min"], 0.32) and approx(b["y_max"], 0.48),
            f"像素坐标换算正确，映射回原图 {b} ≈ (0.42,0.32,0.58,0.48)",
        )


# --------------------------------------------------------------------------- #
# 4. 阈值模式（旧行为）仍然工作
# --------------------------------------------------------------------------- #
def test_threshold_mode(img_dir, out_dir):
    print("[test] zoom_mode=threshold：低置信候选放大精修（旧行为）")
    global SCRIPT
    img = os.path.join(img_dir, "d.png")
    make_image(img, color=(120, 200, 120))

    SCRIPT = {
        0: {
            "objects": [
                {"label": "red circle", "confidence": 0.25,
                 "x_min": 0.4, "y_min": 0.3, "x_max": 0.6, "y_max": 0.5}
            ],
            "zoom_regions": [],  # 模型没有主动请求
        },
        1: {
            "objects": [
                {"label": "red circle", "confidence": 0.9,
                 "x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}
            ],
            "zoom_regions": [],
        },
    }

    cfg = Config(
        task_type="box", targets=["red circle"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2,
        zoom_on_uncertain=True, zoom_via_mcp=True, zoom_mode="threshold",
        zoom_threshold=0.3, zoom_max_levels=1, zoom_upscale=2,
    )

    res = run_annotate(img, cfg)
    objs = res.result.get("objects", [])
    check(len(objs) == 1, "重合的同类检测被合并为 1 个框")
    b = objs[0]
    check(
        approx(b["x_min"], 0.42) and approx(b["x_max"], 0.58)
        and approx(b["y_min"], 0.32) and approx(b["y_max"], 0.48),
        f"精修结果替换原候选 {b} ≈ (0.42,0.32,0.58,0.48)",
    )
    check(b["confidence"] == 0.9, f"置信度提升到 {b['confidence']}")


# --------------------------------------------------------------------------- #
# 5. 模型没请求 → 不产生多余裁切
# --------------------------------------------------------------------------- #
def test_no_zoom_without_model_request(img_dir, out_dir):
    print("[test] zoom_mode=model 且模型未声明 zoom_regions → 不放大")
    global SCRIPT
    img = os.path.join(img_dir, "e.png")
    make_image(img, color=(200, 200, 90))

    SCRIPT = {
        0: {
            "objects": [
                {"label": "red circle", "confidence": 0.25,
                 "x_min": 0.4, "y_min": 0.3, "x_max": 0.6, "y_max": 0.5}
            ],
            "zoom_regions": [],
        },
        1: {"objects": [], "zoom_regions": []},
    }

    cfg = Config(
        task_type="box", targets=["red circle"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2,
        zoom_on_uncertain=True, zoom_via_mcp=True, zoom_mode="model",
        zoom_threshold=0.3, zoom_max_levels=1,
    )

    res = run_annotate(img, cfg)
    objs = res.result.get("objects", [])
    check(len(objs) == 1, "保留原检测 1 个框")
    check(approx(objs[0]["x_min"], 0.4) and approx(objs[0]["x_max"], 0.6),
          f"坐标保持原样（未触发放大）{objs[0]}")
    zoom_root = os.path.join(out_dir, "_zoom")
    produced = list(_walk_png(zoom_root)) if os.path.isdir(zoom_root) else []
    check(len(produced) == 0, f"没有生成任何裁切产物（实际 {len(produced)}）")


def _walk_png(root):
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".png"):
                yield os.path.join(dirpath, f)


# --------------------------------------------------------------------------- #
# 6. harness verify 二次复核
# --------------------------------------------------------------------------- #
def test_harness_verify(img_dir, out_dir):
    print("[test] harness 二次验证（verify）反投影")
    global SCRIPT
    img = os.path.join(img_dir, "f.png")
    make_image(img, color=(150, 120, 220))

    SCRIPT = {
        0: {
            "objects": [
                {"label": "red circle", "confidence": 0.8,
                 "x_min": 0.4, "y_min": 0.3, "x_max": 0.6, "y_max": 0.5}
            ],
            "zoom_regions": [],
        },
        "verify": {"present": True, "label": "red circle", "confidence": 0.9,
                   "x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9},
    }

    cfg = Config(
        task_type="box", targets=["red circle"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2,
        verify=True, verify_rounds=1, verify_margin=0.25, verify_threshold=0.5,
    )

    res = run_annotate(img, cfg)
    objs = res.result.get("objects", [])
    check(len(objs) == 1, "二次验证保留 1 个框")
    b = objs[0]
    # margin=0.25 → 裁切 (0.35,0.25,0.65,0.55)；内部 (0.1,0.1,0.9,0.9)
    # → 原图 (0.38,0.28,0.62,0.52)
    check(
        approx(b["x_min"], 0.38, 1e-3) and approx(b["x_max"], 0.62, 1e-3)
        and approx(b["y_min"], 0.28, 1e-3) and approx(b["y_max"], 0.52, 1e-3),
        f"verify 反投影 {b} ≈ (0.38,0.28,0.62,0.52)",
    )


# --------------------------------------------------------------------------- #
# 7. 自动可视化
# --------------------------------------------------------------------------- #
def test_visualize_default_dir(img_dir, out_dir):
    print("[test] --visualize：默认输出到 <out>/_viz/<stem>.png")
    global SCRIPT
    img = os.path.join(img_dir, "g.png")
    make_image(img)

    SCRIPT = {
        0: {
            "objects": [
                {"label": "red circle", "confidence": 0.9,
                 "x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}
            ],
            "zoom_regions": [],
        }
    }

    cfg = Config(
        task_type="box", targets=["red circle"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2, visualize=True,
    )

    run_annotate(img, cfg)
    viz = os.path.join(out_dir, "_viz", "g.png")
    check(os.path.isfile(viz), f"生成可视化图 {viz}")
    if os.path.isfile(viz):
        out_img = Image.open(viz)
        check(out_img.size == (1000, 800), f"尺寸与原图一致 {out_img.size}")
        check(_bytes_differ(img, viz), "画面确有绘制内容（与原图不同）")


def test_visualize_custom_dir(img_dir, out_dir):
    print("[test] --viz-dir：自定义可视化输出目录")
    global SCRIPT
    img = os.path.join(img_dir, "h.png")
    make_image(img, color=(90, 90, 220))

    SCRIPT = {
        0: {
            "points": [{"label": "red circle", "confidence": 0.9, "x": 0.5, "y": 0.5}],
            "zoom_regions": [],
        }
    }

    custom = os.path.join(out_dir, "my_preview")
    cfg = Config(
        task_type="point", targets=["red circle"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2, visualize=True, viz_dir=custom,
    )

    run_annotate(img, cfg)
    viz = os.path.join(custom, "h.png")
    check(os.path.isfile(viz), f"生成到自定义目录 {viz}")
    if os.path.isfile(viz):
        check(_bytes_differ(img, viz), "point 标注已绘制（与原图不同）")


def _bytes_differ(path_a, path_b) -> bool:
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        return fa.read() != fb.read()


# --------------------------------------------------------------------------- #
# 8. pipeline.run 多图：客户端整批复用
# --------------------------------------------------------------------------- #
def test_run_multiple_images(img_dir, out_dir):
    print("[test] pipeline.run 多图 + 模型驱动放大")
    global SCRIPT
    make_image(os.path.join(img_dir, "i1.png"))
    make_image(os.path.join(img_dir, "i2.png"), color=(90, 90, 220))

    SCRIPT = {
        0: {
            "objects": [],
            "zoom_regions": [{"x_min": 0.4, "y_min": 0.3, "x_max": 0.6, "y_max": 0.5}],
        },
        1: {
            "objects": [
                {"label": "red circle", "confidence": 0.9,
                 "x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}
            ],
            "zoom_regions": [],
        },
    }

    cfg = Config(
        task_type="box", targets=["red circle"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2,
        zoom_on_uncertain=True, zoom_via_mcp=True, zoom_mode="model",
        zoom_max_levels=1, zoom_upscale=2,
    ).validate()
    cfg.workers = 2

    orig = pipeline.annotate
    pipeline.annotate = fake_annotate
    try:
        results = pipeline.run(cfg)
    finally:
        pipeline.annotate = orig

    check(len(results) == 2, f"处理 2 张图（实际 {len(results)}）")
    for r in results:
        name = os.path.basename(r.image_path)
        check(r.status == "ok", f"{name} 状态 ok")
        b = r.result.get("objects", [{}])[0]
        check(
            approx(b["x_min"], 0.42) and approx(b["x_max"], 0.58)
            and approx(b["y_min"], 0.32) and approx(b["y_max"], 0.48),
            f"{name} 放大结果映射正确 {b}",
        )


# --------------------------------------------------------------------------- #
def main() -> int:
    base = tempfile.mkdtemp(prefix="harness_test_")

    plan = [
        test_model_driven_box_single_level,
        test_model_driven_box_multilevel,
        test_model_driven_point,
        test_pixel_coord_zoom_regions,
        test_threshold_mode,
        test_no_zoom_without_model_request,
        test_harness_verify,
        test_visualize_default_dir,
        test_visualize_custom_dir,
        test_run_multiple_images,
    ]
    failed = 0
    for fn in plan:
        # 每个用例独立的 image / output 目录，避免相互干扰
        case_img = os.path.join(base, "img_" + fn.__name__)
        case_out = os.path.join(base, "out_" + fn.__name__)
        os.makedirs(case_img, exist_ok=True)
        os.makedirs(case_out, exist_ok=True)
        try:
            fn(case_img, case_out)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"\n✗ {fn.__name__} 失败：{type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()

    print(f"\n==== harness 集成测试：{len(plan) - failed}/{len(plan)} 通过 ====")
    shutil.rmtree(base, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""harness 循环中 MCP 调用的集成测试（用桩模型确定性驱动，不依赖真实 Ollama 推理）。

目的：验证「放大精修（zoom-on-uncertain）」与「二次验证（harness verify）」在
harness 循环里对 MCP 裁剪放大服务的真实调用是否正确，尤其关注：

- annotate_image 是否正确拉起 / 关闭 MCP 子进程（make_zoom_client）；
- zoom_refine 调用 crop_and_upscale + map_box/point_to_original 后，
  局部坐标是否被正确 compose 回原图坐标（box 单级 / 多级、point）；
- harness_verify 的二次裁切复核路径不崩溃且反投影正确；
- pipeline.run 处理多张图时，每张图都正确经历 MCP 生命周期。

桩模型（fake_annotate）按文件名区分「原图」与「裁切图」：
裁切图（路径含 _zoom / _verify）返回高置信度、几乎占满裁切的框/点；
原图返回低置信度初始检测，从而必然触发 zoom / verify。

运行：
    python tests/test_harness_zoom.py
"""

from __future__ import annotations

import os
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
# 桩模型：替代 ollama_client.annotate
# --------------------------------------------------------------------------- #
def fake_annotate(image_path, prompt, schema_class, config, max_attempts=3, client=None):
    name = schema_class.__name__
    # 用「目录结构」而非脆弱的子串判断裁切图：裁切图位于 <out>/_zoom/<stem>/ 或
    # <out>/_verify/<stem>/ 之下；原图路径不含独立的 _zoom/_verify 目录。
    is_crop = (os.sep + "_zoom" + os.sep) in image_path or (
        os.sep + "_verify" + os.sep
    ) in image_path
    if name == "BoxAnnotation":
        if is_crop:
            return {"objects": [
                {"label": "red circle", "confidence": 0.9,
                 "x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}
            ]}
        return {"objects": [
            {"label": "red circle", "confidence": 0.25,
             "x_min": 0.4, "y_min": 0.3, "x_max": 0.6, "y_max": 0.5}
        ]}
    if name == "PointAnnotation":
        if is_crop:
            return {"points": [
                {"label": "red circle", "confidence": 0.9, "x": 0.5, "y": 0.5}
            ]}
        return {"points": [
            {"label": "red circle", "confidence": 0.25, "x": 0.5, "y": 0.5}
        ]}
    if name == "VerifyAnnotation":
        return {"present": True, "label": "red circle", "confidence": 0.9,
                "x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}
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


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #
def test_zoom_box_single_level(img_dir, out_dir):
    print("[test] harness zoom box 单级（走 MCP 子进程）")
    img = os.path.join(img_dir, "a.png")
    make_image(img)
    cfg = Config(
        task_type="box", targets=["red circle"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2,
        zoom_on_uncertain=True, zoom_via_mcp=True,
        zoom_threshold=0.3, zoom_max_levels=1, zoom_upscale=2, zoom_margin=0.25,
    )
    orig = pipeline.annotate
    pipeline.annotate = fake_annotate
    try:
        res = pipeline.annotate_image(img, cfg)
    finally:
        pipeline.annotate = orig

    objs = res.result.get("objects", [])
    check(len(objs) == 1, f"精炼后保留 1 个框（实际 {len(objs)}）")
    b = objs[0]
    # 期望：(0.42,0.32,0.58,0.48)
    check(
        approx(b["x_min"], 0.42) and approx(b["x_max"], 0.58)
        and approx(b["y_min"], 0.32) and approx(b["y_max"], 0.48),
        f"box 映射回原图 {b} ≈ (0.42,0.32,0.58,0.48)",
    )
    check(b["confidence"] == 0.9, f"置信度提升到 {b['confidence']}")


def test_zoom_box_multilevel(img_dir, out_dir):
    print("[test] harness zoom box 多级（compose 回原图）")
    img = os.path.join(img_dir, "b.png")
    make_image(img)
    cfg = Config(
        task_type="box", targets=["red circle"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2,
        zoom_on_uncertain=True, zoom_via_mcp=True,
        zoom_threshold=0.95, zoom_max_levels=2, zoom_upscale=2, zoom_margin=0.25,
    )
    orig = pipeline.annotate
    pipeline.annotate = fake_annotate
    try:
        res = pipeline.annotate_image(img, cfg)
    finally:
        pipeline.annotate = orig

    objs = res.result.get("objects", [])
    check(len(objs) == 1, "多级精炼后保留 1 个框")
    b = objs[0]
    # 两级展开：orig = region.min + local*(region.max-region.min)，逐级 compose
    # 第1级 region={0.4,0.3,0.6,0.5}；第2级 region={0.42,0.32,0.58,0.48}
    # 期望：(0.436,0.336,0.564,0.464)
    check(
        approx(b["x_min"], 0.436, 1e-3) and approx(b["x_max"], 0.564, 1e-3)
        and approx(b["y_min"], 0.336, 1e-3) and approx(b["y_max"], 0.464, 1e-3),
        f"box 多级映射回原图 {b} ≈ (0.436,0.336,0.564,0.464)",
    )


def test_zoom_point_single_level(img_dir, out_dir):
    print("[test] harness zoom point 单级（走 MCP 子进程）")
    img = os.path.join(img_dir, "c.png")
    make_image(img, color=(80, 160, 200))
    cfg = Config(
        task_type="point", targets=["red circle"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2,
        zoom_on_uncertain=True, zoom_via_mcp=True,
        zoom_threshold=0.3, zoom_max_levels=1, zoom_upscale=2, zoom_margin=0.25,
    )
    orig = pipeline.annotate
    pipeline.annotate = fake_annotate
    try:
        res = pipeline.annotate_image(img, cfg)
    finally:
        pipeline.annotate = orig

    pts = res.result.get("points", [])
    check(len(pts) == 1, "精炼后保留 1 个点")
    p = pts[0]
    # 期望：(0.5, 0.5)
    check(approx(p["x"], 0.5) and approx(p["y"], 0.5), f"point 映射回原图 {p} ≈ (0.5,0.5)")


def test_harness_verify(img_dir, out_dir):
    print("[test] harness 二次验证（verify）路径")
    img = os.path.join(img_dir, "d.png")
    make_image(img, color=(120, 200, 120))
    cfg = Config(
        task_type="box", targets=["red circle"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2,
        verify=True, verify_rounds=1, verify_margin=0.25, verify_threshold=0.5,
    )
    orig = pipeline.annotate
    pipeline.annotate = fake_annotate
    try:
        res = pipeline.annotate_image(img, cfg)
    finally:
        pipeline.annotate = orig

    objs = res.result.get("objects", [])
    check(len(objs) == 1, "二次验证保留 1 个框")
    b = objs[0]
    # 反投影：crop_norm = (0.35,0.25,0.65,0.55)，raw=(0.1,0.1,0.9,0.9)
    # 期望：(0.38,0.28,0.62,0.52)
    check(
        approx(b["x_min"], 0.38, 1e-3) and approx(b["x_max"], 0.62, 1e-3)
        and approx(b["y_min"], 0.28, 1e-3) and approx(b["y_max"], 0.52, 1e-3),
        f"verify 反投影 {b} ≈ (0.38,0.28,0.62,0.52)",
    )


def test_run_multiple_images(img_dir, out_dir):
    print("[test] pipeline.run 多图 + MCP 生命周期")
    make_image(os.path.join(img_dir, "e1.png"))
    make_image(os.path.join(img_dir, "e2.png"), color=(90, 90, 220))
    cfg = Config(
        task_type="box", targets=["red circle"],
        image_dir=img_dir, output_dir=out_dir, output_format="json",
        confidence_threshold=0.2,
        zoom_on_uncertain=True, zoom_via_mcp=True,
        zoom_threshold=0.3, zoom_max_levels=1, zoom_upscale=2,
    ).validate()
    orig = pipeline.annotate
    pipeline.annotate = fake_annotate
    try:
        results = pipeline.run(cfg)
    finally:
        pipeline.annotate = orig

    check(len(results) == 2, f"处理 2 张图（实际 {len(results)}）")
    for r in results:
        check(r.status == "ok", f"{os.path.basename(r.image_path)} 状态 ok")
        b = r.result.get("objects", [{}])[0]
        check(
            approx(b["x_min"], 0.42) and approx(b["x_max"], 0.58)
            and approx(b["y_min"], 0.32) and approx(b["y_max"], 0.48),
            f"{os.path.basename(r.image_path)} 坐标正确",
        )


def main() -> int:
    base = tempfile.mkdtemp(prefix="harness_test_")
    img_dir = os.path.join(base, "images")
    out_dir = os.path.join(base, "out")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    plan = [
        test_zoom_box_single_level,
        test_zoom_box_multilevel,
        test_zoom_point_single_level,
        test_harness_verify,
        test_run_multiple_images,
    ]
    failed = 0
    for fn in plan:
        # 每个用例使用独立的 image / output 目录，避免相互干扰
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

    print(f"\n==== harness MCP 集成测试：{len(plan) - failed}/{len(plan)} 通过 ====")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""MCP 裁剪放大服务的协议层单元测试（不依赖 Ollama / 视觉模型）。

覆盖：
- MCPZoomClient 子进程握手（initialize / notifications/initialized）
- crop_and_upscale 单级放大：裁切尺寸 + 原图矩形 compose 正确
- map_box_to_original / map_point_to_original 坐标映射数值正确
- 多级放大：region 逐级 compose 回原图坐标正确
- LocalZoomEngine 与 MCPZoomClient 结果一致
- make_zoom_client 在 zoom_via_mcp=True 且脚本不存在时回退到 LocalZoomEngine；
  zoom_via_mcp=False 时直接返回 LocalZoomEngine

运行（需要 ollama / pydantic / pillow 环境）：
    python tests/test_mcp_zoom.py
"""

from __future__ import annotations

import os
import sys
import tempfile

# 把项目根目录加入 sys.path，便于 import src / mcp_server
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp_server.crop_zoom_server import (  # noqa: E402
    crop_and_upscale,
    map_box_to_original,
    map_point_to_original,
)
from src.config import Config  # noqa: E402
from src.mcp_zoom import (  # noqa: E402
    LocalZoomEngine,
    MCPZoomClient,
    make_zoom_client,
)

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要 Pillow：pip install -r requirements.txt") from exc


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def make_image(path: str, w: int, h: int, color=(128, 128, 128)) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (w, h), color).save(path)


def approx(a, b, tol=1e-4) -> bool:
    return abs(float(a) - float(b)) <= tol


def check(cond, msg):
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  ✓ {msg}")


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #
def test_mcp_handshake_and_health():
    print("[test] MCP 握手 + health")
    client = MCPZoomClient()
    try:
        client.start()
        out = client._call("health", {})
        check(out.get("ok") is True, "health 返回 {'ok': True}")
    finally:
        client.close()


def test_mcp_crop_single_level(tmp):
    print("[test] MCP 单级裁切 + compose 区域")
    img = os.path.join(tmp, "src.png")
    make_image(img, 1000, 800)
    region = {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
    local = (0.4, 0.3, 0.6, 0.5)
    client = MCPZoomClient()
    try:
        client.start()
        data = client.crop_and_upscale(img, region, local, 2, tmp, tag="L1")
        # crop 像素尺寸 = (0.2*1000*2, 0.2*800*2) = (400, 320)
        check(data["size"] == [400, 320], f"裁切尺寸 {data['size']} == [400, 320]")
        # composed 原图矩形应等于 local（因为 region 是全图）
        c = data["region"]
        check(
            approx(c["x_min"], 0.4) and approx(c["x_max"], 0.6)
            and approx(c["y_min"], 0.3) and approx(c["y_max"], 0.5),
            f"compose 区域 {c} == local",
        )
        check(os.path.isfile(data["cropped_image_path"]), "裁切图已落盘")
    finally:
        client.close()


def test_mcp_map_correctness():
    print("[test] map_box / map_point 数值正确")
    region = {"x_min": 0.2, "y_min": 0.1, "x_max": 0.6, "y_max": 0.5}
    box = map_box_to_original(region, 0.5, 0.5, 0.7, 0.9)
    # orig = region.min + local*(region.max-region.min)
    exp = {
        "x_min": 0.2 + 0.5 * 0.4,
        "y_min": 0.1 + 0.5 * 0.4,
        "x_max": 0.2 + 0.7 * 0.4,
        "y_max": 0.1 + 0.9 * 0.4,
    }
    check(
        approx(box["x_min"], exp["x_min"]) and approx(box["x_max"], exp["x_max"])
        and approx(box["y_min"], exp["y_min"]) and approx(box["y_max"], exp["y_max"]),
        f"map_box 结果 {box} == {exp}",
    )
    pt = map_point_to_original(region, 0.25, 0.75)
    check(
        approx(pt["x"], 0.2 + 0.25 * 0.4) and approx(pt["y"], 0.1 + 0.75 * 0.4),
        f"map_point 结果 {pt} == 期望",
    )


def test_mcp_multilevel_compose(tmp):
    print("[test] MCP 多级放大 compose 坐标")
    img = os.path.join(tmp, "src.png")
    make_image(img, 1000, 800)

    client = MCPZoomClient()
    try:
        client.start()
        region0 = {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
        local0 = (0.4, 0.3, 0.6, 0.5)
        d1 = client.crop_and_upscale(img, region0, local0, 2, tmp, tag="L1")
        composed1 = d1["region"]
        # 第二级：在裁切图内取子矩形 local1，相对裁切图
        local1 = (0.1, 0.1, 0.9, 0.9)
        d2 = client.crop_and_upscale(d1["cropped_image_path"], composed1, local1, 2, tmp, tag="L2")
        composed2 = d2["region"]
        # 期望 composed2 == 两级展开后映射回原图：
        # 第 1 级裁切在原图中的矩形 = map_box_to_original(region0, local0)
        # 第 2 级子框映射回原图 = map_box_to_original(第1级矩形, local1)
        crop_region = map_box_to_original(region0, local0[0], local0[1], local0[2], local0[3])
        expected = map_box_to_original(crop_region, local1[0], local1[1], local1[2], local1[3])
        check(
            approx(composed2["x_min"], expected["x_min"], 1e-3)
            and approx(composed2["x_max"], expected["x_max"], 1e-3)
            and approx(composed2["y_min"], expected["y_min"], 1e-3)
            and approx(composed2["y_max"], expected["y_max"], 1e-3),
            f"二级 compose {composed2} == 期望 {expected}",
        )
    finally:
        client.close()


def test_local_equals_mcp(tmp):
    print("[test] LocalZoomEngine 与 MCPZoomClient 等价")
    img = os.path.join(tmp, "src.png")
    make_image(img, 800, 600)
    region = {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
    local = (0.2, 0.2, 0.8, 0.7)

    local_engine = LocalZoomEngine()
    mcp = MCPZoomClient()
    try:
        mcp.start()
        dl = local_engine.crop_and_upscale(img, region, local, 3, tmp, tag="loc")
        dm = mcp.crop_and_upscale(img, region, local, 3, tmp, tag="mcp")
        check(dl["region"] == dm["region"], "crop region 一致")
        check(dl["size"] == dm["size"], "crop size 一致")
        check(dl["cropped_image_path"] != dm["cropped_image_path"], "落盘路径不同（标签区分）")

        bl = local_engine.map_box_to_original(region, (0.3, 0.3, 0.6, 0.6))
        bm = mcp.map_box_to_original(region, (0.3, 0.3, 0.6, 0.6))
        check(bl == bm, "map_box 一致")
        pl = local_engine.map_point_to_original(region, 0.5, 0.5)
        pm = mcp.map_point_to_original(region, 0.5, 0.5)
        check(pl == pm, "map_point 一致")
    finally:
        mcp.close()
        local_engine.close()


def test_make_zoom_client_fallback():
    print("[test] make_zoom_client 回退逻辑")
    # zoom_via_mcp=False -> LocalZoomEngine
    cfg = Config(task_type="box", targets=["x"], zoom_via_mcp=False)
    eng = make_zoom_client(cfg)
    check(isinstance(eng, LocalZoomEngine), "zoom_via_mcp=False 返回 LocalZoomEngine")

    # zoom_via_mcp=True 但脚本不存在 -> 回退 LocalZoomEngine + 告警
    cfg2 = Config(task_type="box", targets=["x"], zoom_via_mcp=True, zoom_server="__nope__.py")
    eng2 = make_zoom_client(cfg2)
    check(isinstance(eng2, LocalZoomEngine), "脚本不存在时回退 LocalZoomEngine")
    eng2.close()

    # zoom_via_mcp=True 且脚本存在 -> MCPZoomClient（握手成功）
    cfg3 = Config(task_type="box", targets=["x"], zoom_via_mcp=True)
    eng3 = make_zoom_client(cfg3)
    try:
        check(isinstance(eng3, MCPZoomClient), "脚本存在时返回 MCPZoomClient")
        eng3.start()
        check(eng3._call("health", {}).get("ok") is True, "MCP 握手后可调用 health")
    finally:
        eng3.close()


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="mcp_test_")
    plan = [
        (test_mcp_handshake_and_health, ()),
        (test_mcp_crop_single_level, (tmp,)),
        (test_mcp_map_correctness, ()),
        (test_mcp_multilevel_compose, (tmp,)),
        (test_local_equals_mcp, (tmp,)),
        (test_make_zoom_client_fallback, ()),
    ]
    failed = 0
    for fn, args in plan:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"\n✗ {fn.__name__} 失败：{type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n==== MCP 协议测试：{len(plan) - failed}/{len(plan)} 通过 ====")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

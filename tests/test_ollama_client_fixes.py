# -*- coding: utf-8 -*-
"""针对本机 qwen3.5:4b 实测经验提炼的修复测试。

覆盖：
1. 0~1000 归一化坐标识别（大图不再被误当像素缩小约 3 倍）；
2. 小图像素坐标仍兼容（不破坏旧行为）；
3. num_ctx 按图片面积自适应；
4. parse_content 兼容单元素数组。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ollama_client import decide_num_ctx, parse_content
from src.validator import clean, detect_coord_scale

PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {msg}")
    else:
        FAIL += 1
        print(f"  ✗ 失败：{msg}")


def test_detect_scale():
    print("[test] 坐标尺度检测")
    # 大图 + 0~1000 归一化 -> norm1000
    check(
        detect_coord_scale([365, 634, 617, 758], (2945, 1687)) == "norm1000",
        "大图 2945x1687 的 [365,634,617,758] 判为 norm1000",
    )
    # 已是 0~1 -> unit
    check(
        detect_coord_scale([0.12, 0.34, 0.56, 0.78], (2945, 1687)) == "unit",
        "0~1 坐标判为 unit",
    )
    # 小图像素坐标 -> pixel
    check(
        detect_coord_scale([400, 240, 600, 400], (1000, 800)) == "pixel",
        "小图 1000x800 的像素坐标判为 pixel",
    )
    # 大图但值超过 1000 -> 真像素
    check(
        detect_coord_scale([1200, 800, 2000, 1500], (2945, 1687)) == "pixel",
        "大图上超过 1000 的值判为 pixel",
    )


def test_clean_0_1000():
    print("[test] 0~1000 坐标清洗回归")
    raw = {
        "objects": [
            {
                "label": "开始游戏",
                "confidence": 0.95,
                "x_min": 365,
                "y_min": 634,
                "x_max": 617,
                "y_max": 758,
            }
        ]
    }
    result = clean(
        raw,
        "box",
        targets=["开始游戏"],
        image_size=(2945, 1687),
    )
    obj = result["objects"][0]
    check(abs(obj["x_min"] - 0.365) < 1e-6, f"x_min 应为 0.365，实际 {obj['x_min']}")
    check(abs(obj["y_max"] - 0.758) < 1e-6, f"y_max 应为 0.758，实际 {obj['y_max']}")


def test_clean_pixel_unchanged():
    print("[test] 小图像素坐标清洗（旧行为兼容）")
    raw = {
        "objects": [
            {
                "label": "circle",
                "confidence": 0.9,
                "x_min": 400,
                "y_min": 240,
                "x_max": 600,
                "y_max": 400,
            }
        ]
    }
    result = clean(raw, "box", targets=["circle"], image_size=(1000, 800))
    obj = result["objects"][0]
    check(abs(obj["x_min"] - 0.4) < 1e-6, f"x_min 应为 0.4，实际 {obj['x_min']}")
    check(abs(obj["y_min"] - 0.3) < 1e-6, f"y_min 应为 0.3，实际 {obj['y_min']}")


def test_num_ctx():
    print("[test] num_ctx 自适应")
    check(decide_num_ctx("nonexistent.png") == 4096, "无图时回退 4096")
    check(decide_num_ctx("nonexistent.png", requested=8192) == 8192, "显式值被尊重")
    # 造一张大图
    try:
        from PIL import Image

        import tempfile

        p = Path(tempfile.gettempdir()) / "big_img_ctx_test.png"
        Image.new("RGB", (2945, 1687)).save(p)
        ctx = decide_num_ctx(str(p))
        check(ctx > 4096, f"大图 num_ctx 自适应应 >4096，实际 {ctx}")
        p.unlink(missing_ok=True)
    except ImportError:
        print("  - 跳过（无 Pillow）")


def test_parse_array():
    print("[test] parse_content 兼容数组")
    d = parse_content('[{"objects": [], "zoom_regions": []}]')
    check("objects" in d, "单元素数组被拆包为对象")
    d2 = parse_content('```json\n{"summary": "hi"}\n```')
    check(d2.get("summary") == "hi", "代码块包裹对象正常解析")


if __name__ == "__main__":
    for fn in (
        test_detect_scale,
        test_clean_0_1000,
        test_clean_pixel_unchanged,
        test_num_ctx,
        test_parse_array,
    ):
        fn()
    print(f"\n==== 修复测试：{PASS}/{PASS + FAIL} 通过 ====")
    sys.exit(0 if FAIL == 0 else 1)

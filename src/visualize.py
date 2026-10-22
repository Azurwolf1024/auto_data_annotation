# -*- coding: utf-8 -*-
"""可视化模块：把标注结果画回原图，生成带框/点/文字框的 PNG 供人工检查。

对 box 画矩形 + 标签 + 置信度；对 point 画圆点 + 十字 + 标签；对 ocr 画
文字块矩形 + 文字片段。标签按名称稳定配色，文字带黑底以保证在亮/暗图上可读。
caption / tag 没有几何信息，仅原样复制原图以便对照。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional


def _load_font(size: int):
    """尽量加载能显示中文的字体；失败则用 PIL 默认点阵字体。"""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsum.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                from PIL import ImageFont

                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                continue
    try:
        from PIL import ImageDraw

        return ImageDraw.load_default()
    except Exception:  # noqa: BLE001
        return None


_PALETTE = [
    (220, 40, 40),
    (40, 120, 220),
    (40, 170, 90),
    (230, 160, 30),
    (170, 60, 200),
    (20, 190, 190),
    (230, 90, 140),
    (110, 110, 110),
]


def _color_for(label: str):
    """按标签名稳定地选一个高对比颜色（RGB 元组，0~255）。"""
    h = 0
    for ch in label:
        h = (h * 31 + ord(ch)) & 0xFFFF
    return _PALETTE[h % len(_PALETTE)]


def _draw_label(draw, text: str, x, y, color, font, img_width: int) -> None:
    """在 (x, y) 处画带黑底的文字标签，保证在亮/暗图上都可读。"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:  # noqa: BLE001 - 旧版 PIL 无 textbbox
        tw, th = max(10, len(text) * 8), 14
    tx = max(0, min(x, img_width - tw - 2))
    ty = max(0, y - th - 2)
    draw.rectangle([tx, ty, tx + tw + 2, ty + th + 2], fill=(0, 0, 0))
    draw.text((tx + 1, ty + 1), text, fill=color, font=font)


def draw_result(
    image_path: str,
    result: Dict,
    task_type: str,
    output_path: str,
    targets: Optional[List[str]] = None,
) -> str:
    """把 result 画到 image_path 上并保存到 output_path，返回输出路径。

    task_type 支持 box / point / ocr；caption / tag 无几何信息，仅复制原图。
    """
    from PIL import Image, ImageDraw

    if task_type not in ("box", "point", "ocr"):
        img = Image.open(image_path).convert("RGB")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img.save(output_path)
        return output_path

    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)
    font = _load_font(max(12, int(min(width, height) * 0.025)))

    def _txt(label, conf=None):
        return f"{label} {conf:.2f}" if conf is not None else label

    if task_type == "box":
        for obj in result.get("objects", []):
            x0 = obj["x_min"] * width
            y0 = obj["y_min"] * height
            x1 = obj["x_max"] * width
            y1 = obj["y_max"] * height
            color = _color_for(obj.get("label", "?"))
            lw = max(1, int(min(width, height) * 0.006))
            draw.rectangle([x0, y0, x1, y1], outline=color, width=lw)
            _draw_label(draw, _txt(obj.get("label", ""), obj.get("confidence")), x0, y0, color, font, width)
    elif task_type == "point":
        for pt in result.get("points", []):
            px, py = pt["x"] * width, pt["y"] * height
            color = _color_for(pt.get("label", "?"))
            r = max(4, int(min(width, height) * 0.012))
            draw.ellipse([px - r, py - r, px + r, py + r], outline=color, width=max(1, r // 3))
            draw.line([px - r * 2, py, px + r * 2, py], fill=color, width=max(1, r // 4))
            draw.line([px, py - r * 2, px, py + r * 2], fill=color, width=max(1, r // 4))
            _draw_label(draw, _txt(pt.get("label", ""), pt.get("confidence")), px + r, py - r, color, font, width)
    elif task_type == "ocr":
        for blk in result.get("blocks", []):
            x0 = blk["x_min"] * width
            y0 = blk["y_min"] * height
            x1 = blk["x_max"] * width
            y1 = blk["y_max"] * height
            color = _color_for(blk.get("type", "other"))
            lw = max(1, int(min(width, height) * 0.004))
            draw.rectangle([x0, y0, x1, y1], outline=color, width=lw)
            snippet = (blk.get("text", "") or "")[:16].replace("\n", " ")
            _draw_label(draw, snippet, x0, y0, color, font, width)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    return output_path


__all__ = ["draw_result"]

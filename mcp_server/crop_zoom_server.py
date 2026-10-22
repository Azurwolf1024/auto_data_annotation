# -*- coding: utf-8 -*-
"""轻量级 MCP 服务器：裁剪 + 放大工具（stdio / JSON-RPC 2.0，零额外依赖）。

设计目标
--------
- 仅用 Python 标准库实现一个最小可用的 MCP（Model Context Protocol）服务，
  通过 stdio 与任何 MCP 宿主（Claude Desktop / 自建 Agent 等）通信。
- 暴露的核心能力：把图片中某一「归一化区域」裁切并放大，返回裁切图路径，
  以及该裁切图对应的「原图归一化矩形 region」。下游可把裁切图再次喂给视觉
  模型做精细化标注，再用 map_* 工具把局部坐标映射回原图坐标。
- 支持多级放大：region 记录「当前图片所代表的原图矩形」，每次放大都是在上
  一级裁切图里取一个 local 子矩形，compose 出更深一层的 region，从而把
  local 坐标一路映射回最初的原图。

运行方式
--------
    python -m mcp_server.crop_zoom_server
    python mcp_server/crop_zoom_server.py

协议要点
--------
- 每行一个 JSON 对象（newline-delimited JSON）。
- 支持方法：initialize / notifications/initialized / tools/list / tools/call / ping。
- tools/call 的 result.content[0].text 内是结果的 JSON 字符串。
- 仅 stdout 输出协议 JSON；其它日志一律写 stderr，避免污染协议流。
"""

from __future__ import annotations

import json
import os
import sys
import uuid

PROTOCOL_VERSION = "2024-11-05"


# --------------------------------------------------------------------------- #
# 工具实现（纯函数，既供 MCP tools/call 调用，也可被 pipeline 直接 import 复用）
# --------------------------------------------------------------------------- #
def clamp(v: float) -> float:
    """把值收敛到 [0, 1]。"""
    return max(0.0, min(1.0, float(v)))


def _region_of(region) -> tuple:
    """从各种矩形表示里取出归一化四角 (x_min, y_min, x_max, y_max)。

    兼容三种常见写法，避免调用方/传输格式差异导致失败：
    1. dict：{"x_min", "y_min", "x_max", "y_max"}（MCP JSON 对象）；
    2. 长度为 4 的数组/元组：[x_min, y_min, x_max, y_max]
       （注意：JSON 传输会把 tuple 序列化成 list，故必须支持）；
    3. 带同名属性的对象（进程内直接调用时常见）。
    """
    if isinstance(region, dict):
        return (
            float(region["x_min"]),
            float(region["y_min"]),
            float(region["x_max"]),
            float(region["y_max"]),
        )
    if isinstance(region, (list, tuple)) and len(region) == 4:
        return tuple(float(v) for v in region)
    return float(region.x_min), float(region.y_min), float(region.x_max), float(region.y_max)


def map_point_to_original(region, x: float, y: float) -> dict:
    """把相对某裁切图的归一化点 (x,y) 映射回原图归一化坐标。

    orig = region.min + local * (region.max - region.min)
    """
    rx0, ry0, rx1, ry1 = _region_of(region)
    return {
        "x": rx0 + clamp(x) * (rx1 - rx0),
        "y": ry0 + clamp(y) * (ry1 - ry0),
    }


def map_box_to_original(region, x_min: float, y_min: float, x_max: float, y_max: float) -> dict:
    """把相对某裁切图的归一化框映射回原图归一化坐标。"""
    rx0, ry0, rx1, ry1 = _region_of(region)
    return {
        "x_min": rx0 + clamp(x_min) * (rx1 - rx0),
        "y_min": ry0 + clamp(y_min) * (ry1 - ry0),
        "x_max": rx0 + clamp(x_max) * (rx1 - rx0),
        "y_max": ry0 + clamp(y_max) * (ry1 - ry0),
    }


def crop_and_upscale(
    image_path: str,
    region,
    local: tuple,
    upscale: int = 2,
    output_dir: str = "",
    tag: str = "",
) -> dict:
    """裁切并放大图片中某一区域，返回裁切图路径与 compose 后的原图矩形。

    参数
    ----
    image_path : 当前图片路径（原图或上一级裁切图）。
    region     : 当前图片所代表的原图归一化矩形 {x_min,y_min,x_max,y_max}；
                 原图应为 {"x_min":0,"y_min":0,"x_max":1,"y_max":1}。
    local      : 相对当前图片的裁切子矩形 (x_min,y_min,x_max,y_max)，均在 0~1。
    upscale    : 放大倍数（像素维度）。
    output_dir : 裁切图保存目录。
    tag        : 文件名标签（用于人工核对时区分层级/候选）。

    返回
    ----
    {
      "cropped_image_path": str,          # 裁切放大后的图片路径
      "region": {x_min,y_min,x_max,y_max},# 该裁切图对应的【原图】归一化矩形（已 compose）
      "upscale": int,
      "size": [w, h],                      # 裁切图像素尺寸
      "note": str,
    }
    """
    from PIL import Image

    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), "_zoom")
    os.makedirs(output_dir, exist_ok=True)

    lx0, ly0, lx1, ly1 = (clamp(v) for v in local)
    # compose：当前裁切子矩形在原图中代表的矩形
    composed = map_box_to_original(region, lx0, ly0, lx1, ly1)

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    px0 = max(0, min(int(round(lx0 * W)), W))
    py0 = max(0, min(int(round(ly0 * H)), H))
    px1 = max(0, min(int(round(lx1 * W)), W))
    py1 = max(0, min(int(round(ly1 * H)), H))
    if px1 <= px0:
        px1 = min(px0 + 1, W)
    if py1 <= py0:
        py1 = min(py0 + 1, H)

    crop = img.crop((px0, py0, px1, py1))
    crop = crop.resize(
        (max(1, (px1 - px0) * upscale), max(1, (py1 - py0) * upscale)),
        Image.LANCZOS,
    )

    name = f"crop_{tag}.png" if tag else f"crop_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(output_dir, name)
    crop.save(out_path)

    return {
        "cropped_image_path": out_path,
        "region": composed,
        "upscale": int(upscale),
        "size": [crop.width, crop.height],
        "note": (
            "裁切图坐标系：crop-local 归一化坐标经 region 映射回原图，"
            "即 orig = region.min + local * (region.max - region.min)。"
            "多级放大时把本裁切图作为新 image_path、composed region 作为新 region 继续调用即可。"
        ),
    }


# --------------------------------------------------------------------------- #
# 工具清单（供 tools/list 返回）
# --------------------------------------------------------------------------- #
_REGION_SCHEMA = {
    "type": "object",
    "description": "归一化矩形 {x_min,y_min,x_max,y_max}，坐标范围 0~1；原图即 {0,0,1,1}",
    "properties": {
        "x_min": {"type": "number"},
        "y_min": {"type": "number"},
        "x_max": {"type": "number"},
        "y_max": {"type": "number"},
    },
    "required": ["x_min", "y_min", "x_max", "y_max"],
}

TOOLS = [
    {
        "name": "crop_and_upscale",
        "description": (
            "把图片中某一归一化区域裁切并放大，返回裁切图路径与『该裁切图对应的原图归一化矩形 "
            "region』。支持多级放大：传入当前图片路径与它在原图中所代表的 region，再给一个相对"
            "当前图片的 local 子矩形，即可继续放大更深一层（递归放大）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "当前待裁切的图片路径（原图或上一级裁切图）",
                },
                "region": _REGION_SCHEMA,
                "local_x_min": {"type": "number", "description": "裁切子区域左上 x，相对当前图片归一化 0~1"},
                "local_y_min": {"type": "number", "description": "裁切子区域左上 y，相对当前图片归一化 0~1"},
                "local_x_max": {"type": "number", "description": "裁切子区域右下 x，相对当前图片归一化 0~1"},
                "local_y_max": {"type": "number", "description": "裁切子区域右下 y，相对当前图片归一化 0~1"},
                "upscale": {"type": "integer", "description": "放大倍数（像素维度），默认 2", "default": 2},
                "output_dir": {"type": "string", "description": "裁切图保存目录"},
                "tag": {"type": "string", "description": "文件名标签（用于人工核对），可选"},
            },
            "required": [
                "image_path",
                "region",
                "local_x_min",
                "local_y_min",
                "local_x_max",
                "local_y_max",
            ],
        },
    },
    {
        "name": "map_box_to_original",
        "description": "把一个相对某裁切图的归一化框映射回原图归一化坐标。orig = region.min + local*(region.max-region.min)。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "region": _REGION_SCHEMA,
                "x_min": {"type": "number"},
                "y_min": {"type": "number"},
                "x_max": {"type": "number"},
                "y_max": {"type": "number"},
            },
            "required": ["region", "x_min", "y_min", "x_max", "y_max"],
        },
    },
    {
        "name": "map_point_to_original",
        "description": "把一个相对某裁切图的点映射回原图归一化坐标。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "region": _REGION_SCHEMA,
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
            "required": ["region", "x", "y"],
        },
    },
    {
        "name": "health",
        "description": "健康检查，返回 {'ok': True}。",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# --------------------------------------------------------------------------- #
# 工具分发
# --------------------------------------------------------------------------- #
def dispatch(name: str, args: dict) -> dict:
    if name == "crop_and_upscale":
        region = args.get("region")
        local = (
            float(args["local_x_min"]),
            float(args["local_y_min"]),
            float(args["local_x_max"]),
            float(args["local_y_max"]),
        )
        return crop_and_upscale(
            image_path=str(args["image_path"]),
            region=region,
            local=local,
            upscale=int(args.get("upscale", 2) or 2),
            output_dir=str(args.get("output_dir", "") or ""),
            tag=str(args.get("tag", "") or ""),
        )
    if name == "map_box_to_original":
        return map_box_to_original(
            args["region"],
            float(args["x_min"]),
            float(args["y_min"]),
            float(args["x_max"]),
            float(args["y_max"]),
        )
    if name == "map_point_to_original":
        return map_point_to_original(args["region"], float(args["x"]), float(args["y"]))
    if name == "health":
        return {"ok": True}
    raise ValueError(f"未知工具：{name}")


# --------------------------------------------------------------------------- #
# stdio JSON-RPC 2.0 事件循环
# --------------------------------------------------------------------------- #
def _write(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _read_line() -> str:
    line = sys.stdin.readline()
    if not line:
        return ""
    return line.strip()


def serve() -> None:
    """阻塞式 stdio 服务循环。"""
    while True:
        line = _read_line()
        if not line:
            break  # stdin 关闭，退出
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = req.get("method")
        msg_id = req.get("id")
        params = req.get("params") or {}

        # 通知（无 id）只应答握手，不作业务处理
        if method == "notifications/initialized":
            continue

        if method == "initialize":
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "crop-zoom", "version": "1.0.0"},
            }
            _write({"jsonrpc": "2.0", "id": msg_id, "result": result})
            continue

        if method == "tools/list":
            _write({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}})
            continue

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            try:
                out = dispatch(name, arguments)
                _write(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [
                                {"type": "text", "text": json.dumps(out, ensure_ascii=False)}
                            ],
                            "isError": False,
                        },
                    }
                )
            except Exception as exc:  # noqa: BLE001 - 把错误原样返回给调用方
                _write(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                            "isError": True,
                        },
                    }
                )
            continue

        if method == "ping":
            _write({"jsonrpc": "2.0", "id": msg_id, "result": {}})
            continue

        # 其它方法：返回方法未找到
        _write(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"unknown method: {method}"},
            }
        )


def main() -> None:
    try:
        serve()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

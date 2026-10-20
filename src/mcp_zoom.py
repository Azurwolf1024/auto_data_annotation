# -*- coding: utf-8 -*-
"""放大精修客户端：把 MCP 裁剪放大服务接入标注流水线。

提供两类实现，统一接口（crop_and_upscale / map_box_to_original /
map_point_to_original / close）：

1. LocalZoomEngine —— 直接 import 服务器模块里的纯函数调用，无子进程、最快，
   适合流水线内部重复使用（不依赖外部 MCP 宿主）。
2. MCPZoomClient    —— 以子进程方式拉起 crop_zoom_server.py，走标准 MCP
   stdio 握手（initialize -> tools/list -> tools/call），真正「通过 MCP 工具
   调用」完成放大与坐标映射。供想把它挂到任意 MCP 宿主（如 Claude Desktop）时
   使用，也用于验证服务端协议正确。

默认流水线使用 MCPZoomClient（config.zoom_via_mcp=True）；若启动失败会自动
回退到 LocalZoomEngine 并告警。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, Tuple

# 项目根目录（含 mcp_server/ 与 src/）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _region(x_min: float, y_min: float, x_max: float, y_max: float) -> Dict[str, float]:
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


# --------------------------------------------------------------------------- #
# 本地引擎：直接复用服务器纯函数（零 IPC）
# --------------------------------------------------------------------------- #
class LocalZoomEngine:
    """直接 import 服务器模块函数，无子进程。"""

    def __init__(self):
        sys.path.insert(0, _PROJECT_ROOT)
        from mcp_server.crop_zoom_server import (  # noqa: WPS433 - 运行时导入
            crop_and_upscale,
            map_box_to_original,
            map_point_to_original,
        )

        self._crop = crop_and_upscale
        self._map_box = map_box_to_original
        self._map_point = map_point_to_original

    def crop_and_upscale(
        self,
        image_path: str,
        region: Dict[str, float],
        local: Tuple[float, float, float, float],
        upscale: int,
        output_dir: str,
        tag: str = "",
    ) -> Dict:
        return self._crop(image_path, region, local, upscale, output_dir, tag)

    def map_box_to_original(self, region: Dict[str, float], box: Tuple[float, float, float, float]) -> Dict:
        return self._map_box(region, *box)

    def map_point_to_original(self, region: Dict[str, float], x: float, y: float) -> Dict:
        return self._map_point(region, x, y)

    def close(self) -> None:  # pragma: no cover - 无资源需要释放
        pass


# --------------------------------------------------------------------------- #
# MCP stdio 客户端：子进程 + JSON-RPC 握手
# --------------------------------------------------------------------------- #
class MCPZoomClient:
    """以子进程方式连接 crop_zoom_server.py，走标准 MCP stdio 协议。"""

    def __init__(self, server_script: str = ""):
        if not server_script:
            server_script = os.path.join(_PROJECT_ROOT, "mcp_server", "crop_zoom_server.py")
        self.server_script = os.path.abspath(server_script)
        if not os.path.isfile(self.server_script):
            raise FileNotFoundError(f"MCP 服务器脚本不存在：{self.server_script}")

        self.proc = subprocess.Popen(
            [sys.executable, self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._id = 0
        self._started = False

    # ---- 底层传输 ----
    def _send(self, method: str, params=None, *, notification: bool = False):
        self._id += 1
        msg = {"jsonrpc": "2.0", "method": method}
        if not notification:
            msg["id"] = self._id
        if params is not None:
            msg["params"] = params
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        if notification:
            return None
        return self._recv(self._id)

    def _recv(self, want_id):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in msg and msg["id"] == want_id:
                return msg

    # ---- 握手 ----
    def start(self):
        self._send(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "auto-annotator", "version": "1.0.0"},
            },
        )
        # 通知服务端初始化完成（无需应答）
        self._send("notifications/initialized", {}, notification=True)
        self._started = True
        return True

    def list_tools(self):
        return self._send("tools/list", {})

    def _call(self, name: str, arguments: dict) -> dict:
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        if resp is None:
            raise RuntimeError("MCP 服务无响应（可能已退出）")
        if "error" in resp:
            raise RuntimeError(f"MCP 错误：{resp['error']}")
        result = resp.get("result", {})
        if result.get("isError"):
            text = ""
            for c in result.get("content", []):
                text += c.get("text", "")
            raise RuntimeError(f"工具执行失败：{text}")
        text = ""
        for c in result.get("content", []):
            text += c.get("text", "")
        return json.loads(text)

    # ---- 业务接口 ----
    def crop_and_upscale(
        self,
        image_path: str,
        region: Dict[str, float],
        local: Tuple[float, float, float, float],
        upscale: int,
        output_dir: str,
        tag: str = "",
    ) -> Dict:
        return self._call(
            "crop_and_upscale",
            {
                "image_path": image_path,
                "region": region,
                "local_x_min": local[0],
                "local_y_min": local[1],
                "local_x_max": local[2],
                "local_y_max": local[3],
                "upscale": upscale,
                "output_dir": output_dir,
                "tag": tag,
            },
        )

    def map_box_to_original(self, region: Dict[str, float], box: Tuple[float, float, float, float]) -> Dict:
        return self._call(
            "map_box_to_original",
            {
                "region": region,
                "x_min": box[0],
                "y_min": box[1],
                "x_max": box[2],
                "y_max": box[3],
            },
        )

    def map_point_to_original(self, region: Dict[str, float], x: float, y: float) -> Dict:
        return self._call("map_point_to_original", {"region": region, "x": x, "y": y})

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def make_zoom_client(config) -> "LocalZoomEngine | MCPZoomClient":
    """根据配置创建放大客户端。

    - zoom_via_mcp=True：优先用 MCPZoomClient 连接真实 MCP 服务；失败回退本地引擎。
    - zoom_via_mcp=False：直接用 LocalZoomEngine（无子进程，最快）。
    """
    if getattr(config, "zoom_via_mcp", True):
        try:
            client = MCPZoomClient(getattr(config, "zoom_server", "") or "")
            client.start()
            return client
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ MCP 放大服务启动失败，改用本地引擎：{exc}", flush=True)
    return LocalZoomEngine()


__all__ = ["LocalZoomEngine", "MCPZoomClient", "make_zoom_client"]

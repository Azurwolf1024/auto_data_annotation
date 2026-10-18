# -*- coding: utf-8 -*-
"""Ollama 客户端模块：调用本地视觉模型并返回结构化 dict。

要点：
- 使用官方 `ollama` Python 库的 Client(host=...).chat()；
- 通过 `format` 参数传入 pydantic 模型的 JSON Schema 强制结构化输出；
- options 中设置 temperature=0 保证结果稳定；
- 解析失败自动重试 2 次（共 3 次尝试），仍失败抛 RuntimeError。
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

try:  # pragma: no cover - 依赖未安装时给出清晰报错
    from ollama import Client
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "未安装 ollama Python 库，请先执行：pip install -r requirements.txt"
    ) from exc

MAX_ATTEMPTS = 3  # 首次调用 + 重试 2 次
RETRY_SLEEP = 1.5  # 每次重试前的等待秒数


# --------------------------------------------------------------------------- #
# JSON Schema 处理
# --------------------------------------------------------------------------- #
_DROP_KEYS = {"title", "default", "$defs", "definitions", "additionalProperties"}


def _resolve_refs(node: Any, defs: Dict[str, Any]) -> Any:
    """把 pydantic 生成的 $ref / $defs 结构原地展开，提升 Ollama 兼容性。"""
    if isinstance(node, dict):
        if "$ref" in node:
            ref_name = str(node["$ref"]).split("/")[-1]
            target = copy.deepcopy(defs.get(ref_name, {}))
            target = _resolve_refs(target, defs)
            for key, value in node.items():
                if key != "$ref":
                    target[key] = _resolve_refs(value, defs)
            return target
        return {
            key: _resolve_refs(value, defs)
            for key, value in node.items()
            if key not in _DROP_KEYS or key == "additionalProperties"
        }
    if isinstance(node, list):
        return [_resolve_refs(item, defs) for item in node]
    return node


def simplify_schema(schema_class: Type[BaseModel]) -> Dict[str, Any]:
    """把 pydantic 模型的 JSON Schema 转成 Ollama 友好的扁平 schema。"""
    raw = schema_class.model_json_schema()
    defs = raw.get("$defs", {}) or raw.get("definitions", {}) or {}
    resolved = _resolve_refs(raw, defs)
    resolved.pop("$defs", None)
    resolved.pop("definitions", None)
    resolved.setdefault("type", "object")
    if "properties" in resolved:
        resolved.setdefault("required", list(resolved["properties"].keys()))
        resolved["additionalProperties"] = False
    return resolved


# --------------------------------------------------------------------------- #
# 内容解析
# --------------------------------------------------------------------------- #
def _strip_code_fence(text: str) -> str:
    """去掉模型可能附加的 ```json ... ``` 代码块标记。"""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -3]
    return text.strip()


def _extract_json_object(text: str) -> str:
    """从文本中截取第一个完整的 JSON 对象（大括号配对）。"""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:]


def parse_content(content: str) -> Dict[str, Any]:
    """把模型返回的文本解析成 dict，失败抛 ValueError。"""
    if content is None:
        raise ValueError("模型返回内容为空")
    text = _extract_json_object(_strip_code_fence(content))
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"模型返回的不是 JSON 对象：{type(data).__name__}")
    return data


# --------------------------------------------------------------------------- #
# 主调用
# --------------------------------------------------------------------------- #
_LOCAL_HOST_HINTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def build_client(host: Optional[str] = None):
    """创建 Ollama Client。

    访问本机服务时关闭 httpx 的 trust_env，避免系统 HTTP_PROXY 把
    localhost 请求也转发到代理（会出现 502 upstream connect failed）。
    访问远端服务时保留环境变量代理配置。
    """
    host = host or "http://localhost:11434"
    kwargs: Dict[str, Any] = {}
    lowered = str(host).lower()
    if any(hint in lowered for hint in _LOCAL_HOST_HINTS):
        kwargs["trust_env"] = False
    try:
        return Client(host=host, **kwargs)
    except TypeError:  # 旧版 ollama 库不接受额外参数
        return Client(host=host)


def _call_chat(
    client: "Client",
    model: str,
    prompt: str,
    image_path: str,
    schema: Dict[str, Any],
    think: bool,
) -> Any:
    """发起一次 chat 调用；不支持 think 参数的旧版库会自动降级。"""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_path]}],
        "format": schema,
        "options": {"temperature": 0},
    }
    if not think:
        kwargs["think"] = False
    try:
        return client.chat(**kwargs)
    except TypeError:
        # 旧版本 ollama 库不接受 think 参数
        kwargs.pop("think", None)
        return client.chat(**kwargs)


def annotate(
    image_path: str,
    prompt: str,
    schema_class: Type[BaseModel],
    config,
    max_attempts: int = MAX_ATTEMPTS,
    client: Optional["Client"] = None,
) -> Dict[str, Any]:
    """调用 Ollama 视觉模型标注单张图片。

    参数
    ----
    image_path : 图片路径（传给 ollama 的 images 列表）
    prompt : 中文提示词
    schema_class : pydantic 模型类，用于生成 format schema
    config : src.config.Config 实例（提供 model_name / ollama_url）
    max_attempts : 最大尝试次数，默认 3（首调 + 2 次重试）

    返回
    ----
    dict，结构与 schema_class 一致

    异常
    ----
    RuntimeError：三次尝试全部失败
    """
    schema = simplify_schema(schema_class)
    own_client = client is None
    if own_client:
        client = build_client(getattr(config, "ollama_url", "http://localhost:11434"))

    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        think_flag = attempt == 1  # 第一次尝试关闭思考，重试时放开以增加多样性
        try:
            response = _call_chat(
                client=client,
                model=config.model_name,
                prompt=prompt,
                image_path=str(image_path),
                schema=schema,
                think=think_flag,
            )
            message = getattr(response, "message", None)
            if message is None and isinstance(response, dict):
                message = response.get("message", {})
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            data = parse_content(content or "")
            # 用 schema 校验一次，确保字段齐全（缺字段用默认值补齐）
            validated = schema_class.model_validate(data)
            return validated.model_dump()
        except Exception as exc:  # noqa: BLE001 - 任何异常都要重试
            last_error = exc
            if attempt < max_attempts:
                time.sleep(RETRY_SLEEP * attempt)
                continue
        finally:
            if own_client and attempt == max_attempts:
                pass  # Client 无状态，无需显式关闭

    raise RuntimeError(
        f"标注失败（已尝试 {max_attempts} 次）：{image_path} -> {last_error}"
    )


__all__ = ["annotate", "simplify_schema", "parse_content", "MAX_ATTEMPTS"]

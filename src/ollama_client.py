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

# num_ctx 自适应（本机 qwen3.5:4b 实测）：
#   1024x768 图 -> 约 300~800 图像 token
#   2945x1687 图 -> 约 4090 图像 token（prompt_eval_count 实测）
# 模型默认 num_ctx=4096，大图几乎占满上下文，导致生成被截断到 4~6 token
# 就停下（表现为输出残缺/空）。按像素面积估算图像 token 并留出输出预算，
# 可彻底避免该问题。系数按约 1200 px/token 估（偏保守取整）。
PIXELS_PER_IMAGE_TOKEN = 1200
DEFAULT_NUM_CTX = 4096
CTX_SAFETY_MARGIN = 512  # 给提示词与结构开销留的余量
DEFAULT_OUTPUT_BUDGET = 2048  # num_predict：密集 OCR / 多目标输出可能很长，给足预算


def estimate_image_tokens(image_path: str) -> int:
    """按图片像素面积估算视觉 token 数；读不到尺寸时返回 0。"""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            w, h = img.size
        return max(1, int((w * h) / PIXELS_PER_IMAGE_TOKEN))
    except Exception:  # noqa: BLE001 - 无 Pillow 或损坏图片时静默降级
        return 0


def decide_num_ctx(
    image_path: str,
    requested: int = 0,
    output_budget: int = DEFAULT_OUTPUT_BUDGET,
) -> int:
    """决定本次调用的 num_ctx。

    requested>0 时尊重用户显式设置；否则 = 图像 token + 输出预算 + 余量，
    且不低于 DEFAULT_NUM_CTX。这样小图仍用 4096，大图自动扩到 8192 左右。
    """
    if requested and int(requested) > 0:
        return int(requested)
    img_tokens = estimate_image_tokens(image_path)
    need = img_tokens + int(output_budget) + CTX_SAFETY_MARGIN
    return max(DEFAULT_NUM_CTX, need)


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
    """把模型返回的文本解析成 dict，失败抛 ValueError。

    实测兼容点（qwen3.5:4b）：
    - 有时把对象包成单元素数组 `[{"objects": [...]}]`，这里拆包取第一个 dict；
    - 偶尔在 JSON 前后夹带说明文字，_extract_json_object 负责截取对象主体。
    """
    if content is None:
        raise ValueError("模型返回内容为空")
    text = _extract_json_object(_strip_code_fence(content))
    data = json.loads(text)
    if isinstance(data, list):
        # 模型偶尔返回数组：取其中第一个 dict 元素
        for item in data:
            if isinstance(item, dict):
                data = item
                break
        else:
            raise ValueError("模型返回的数组中没有 JSON 对象")
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
    num_ctx: int = 0,
    num_predict: int = 0,
) -> Any:
    """发起一次 chat 调用；不支持 think 参数的旧版库会自动降级。

    - think=False 时显式关闭思考：结构化标注不需要思考，且实测 qwen3.5:4b
      开启思考会把 token 浪费在推理上、几乎不给最终答案（甚至输出为空）。
    - num_ctx>0 时写入 options，避免大图撑爆默认 4096 上下文导致输出被截断。
    - num_predict>0 时设定输出预算：大图 OCR/密集目标输出很长，不设会被截断成
      半截 JSON（表现为 "Expecting property name ... line N" 解析错误）。
    """
    options: Dict[str, Any] = {"temperature": 0}
    if num_ctx and num_ctx > 0:
        options["num_ctx"] = num_ctx
    if num_predict and num_predict > 0:
        options["num_predict"] = num_predict
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_path]}],
        "format": schema,
        "options": options,
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

    # num_ctx / num_predict 自适应：
    # - 大图必须扩上下文窗，否则图像 token 占满 4096 导致输出被截断到几个 token；
    # - 密集 OCR/多目标输出很长，num_predict 不足会得到半截 JSON 解析失败。
    max_tokens = int(getattr(config, "max_tokens", DEFAULT_OUTPUT_BUDGET) or DEFAULT_OUTPUT_BUDGET)
    num_ctx = decide_num_ctx(
        str(image_path),
        requested=getattr(config, "num_ctx", 0),
        output_budget=max_tokens,
    )
    num_predict = max_tokens

    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        response: Any = None
        # 结构化标注始终关闭思考。实测 qwen3.5:4b 思考模式下会把全部预算花在
        # 推理过程、最终 JSON 为空；只有 think=False 才能稳定给出结构化答案。
        think_flag = False
        try:
            response = _call_chat(
                client=client,
                model=config.model_name,
                prompt=prompt,
                image_path=str(image_path),
                schema=schema,
                think=think_flag,
                num_ctx=num_ctx,
                num_predict=num_predict,
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
            # 若响应携带真实计数，用它们反推下一次的预算（只升不降）：
            # - prompt_eval_count 修正 num_ctx，比像素估算更准，兜住估算偏小的图；
            # - eval_count 已顶到 num_predict 说明输出被截断，翻倍输出预算。
            try:
                used_prompt = getattr(response, "prompt_eval_count", 0) or 0
                used_eval = getattr(response, "eval_count", 0) or 0
                if isinstance(response, dict):
                    used_prompt = response.get("prompt_eval_count", 0) or 0
                    used_eval = response.get("eval_count", 0) or 0
                if used_prompt and int(used_prompt) + num_predict + CTX_SAFETY_MARGIN > num_ctx:
                    num_ctx = int(used_prompt) + num_predict + CTX_SAFETY_MARGIN
                if used_eval and int(used_eval) >= num_predict:
                    num_predict = int(num_predict) * 2
                    num_ctx = max(num_ctx, int(used_prompt) + num_predict + CTX_SAFETY_MARGIN)
            except Exception:  # noqa: BLE001
                pass
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

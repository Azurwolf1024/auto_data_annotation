# -*- coding: utf-8 -*-
"""Prompt 模块：五种任务类型的中文提示词。

所有 prompt 都明确要求：
1. 只输出 JSON，不要任何解释、前后缀或 markdown 代码块；
2. box / point / ocr 的坐标归一化到 0~1。
"""

from __future__ import annotations

from typing import List

from .config import TASK_TYPES

NO_EXPLAIN = (
    "严格要求：只输出一个合法的 JSON 对象，不要输出任何解释说明，"
    "不要输出 markdown 代码块标记（```），不要在 JSON 前后添加任何文字。"
)

NORM_HINT = (
    "坐标必须归一化到 0~1：x 方向除以图片宽度，y 方向除以图片高度，"
    "保留 4 位小数，图片左上角为 (0,0)，右下角为 (1,1)。不要输出像素坐标。"
)


def _format_targets(targets: List[str]) -> str:
    """把候选标签渲染成 prompt 中的列表文本。"""
    if not targets:
        return "（未提供）"
    return "、".join(f'"{t}"' for t in targets)


def build_prompt(task_type: str, targets: List[str] = None) -> str:
    """根据任务类型构造中文 prompt。"""
    targets = list(targets or [])
    task_type = (task_type or "").strip().lower()

    if task_type not in TASK_TYPES:
        raise ValueError(f"未知 task_type：{task_type!r}，可选：{TASK_TYPES}")

    if task_type == "box":
        return (
            "你是一个专业的图像标注助手。请仔细观察这张图片，找出所有可见的目标物体。\n"
            f"候选标签列表：{_format_targets(targets)}\n"
            "规则：\n"
            "1. 每个目标的 label 必须严格从候选标签列表中选择，禁止自创标签；\n"
            "2. 不在候选列表中的物体直接忽略，不要输出；\n"
            "3. confidence 是 0~1 之间的小数，表示你对该目标的置信程度；\n"
            "4. 每个目标给出紧贴物体边缘的矩形框（x_min, y_min, x_max, y_max）；\n"
            f"5. {NORM_HINT}\n"
            f"{NO_EXPLAIN}\n"
            '输出格式：{"objects": [{"label": "类别", "confidence": 0.95, '
            '"x_min": 0.12, "y_min": 0.34, "x_max": 0.56, "y_max": 0.78}]}\n'
            '如果图中没有任何候选目标，输出 {"objects": []}。'
        )

    if task_type == "point":
        return (
            "你是一个专业的图像标注助手。请仔细观察这张图片，找出所有可见的目标物体，"
            "并给出每个物体的中心点坐标。\n"
            f"候选标签列表：{_format_targets(targets)}\n"
            "规则：\n"
            "1. 每个目标的 label 必须严格从候选标签列表中选择，禁止自创标签；\n"
            "2. 不在候选列表中的物体直接忽略，不要输出；\n"
            "3. 只标注物体的几何中心点，不要标注人体关节等结构化关键点；\n"
            "4. confidence 是 0~1 之间的小数，表示你对该目标的置信程度；\n"
            f"5. {NORM_HINT}\n"
            f"{NO_EXPLAIN}\n"
            '输出格式：{"points": [{"label": "类别", "confidence": 0.92, '
            '"x": 0.45, "y": 0.51}]}\n'
            '如果图中没有任何候选目标，输出 {"points": []}。'
        )

    if task_type == "ocr":
        return (
            "你是一个专业的 OCR 文字识别助手。请识别这张图片中出现的全部文字。\n"
            "规则：\n"
            "1. 按文字块（block）组织输出，每个 block 是一段语义完整的文字；\n"
            "2. text 字段为识别到的原文内容，原有换行用空格代替；\n"
            "3. type 字段从以下取值中选择：title、paragraph、table、handwriting、"
            "signage、caption、other；\n"
            "4. 为每个文字块给出其外接矩形框（x_min, y_min, x_max, y_max）；\n"
            f"5. {NORM_HINT}\n"
            "6. 不要遗漏任何文字，包括水印、Logo 上的文字；\n"
            f"{NO_EXPLAIN}\n"
            '输出格式：{"blocks": [{"text": "文字内容", "type": "title", '
            '"x_min": 0.05, "y_min": 0.08, "x_max": 0.90, "y_max": 0.15}]}\n'
            '如果图中没有文字，输出 {"blocks": []}。'
        )

    if task_type == "caption":
        if targets:
            subject = "、".join(str(t) for t in targets)
            focus = (
                f"请围绕以下指定对象进行描述：{subject}。\n"
                "重点描述它们的数量、外观、位置关系、动作状态以及与场景的交互，"
                "不要泛泛描述整张图。"
            )
        else:
            focus = (
                "请描述整张图片：包括场景类型、主要物体、人物或动物的动作、"
                "空间布局、光线与色调氛围。"
            )
        return (
            "你是一个专业的图像描述助手。请仔细观察这张图片并写出中文描述。\n"
            f"{focus}\n"
            "规则：\n"
            "1. 使用简体中文，一到三句话，控制在 120 字以内；\n"
            "2. 描述客观事实，不要臆测图片中不存在的细节；\n"
            "3. 不要出现“这张图片显示了”“图中可以看到”这类冗余开头。\n"
            f"{NO_EXPLAIN}\n"
            '输出格式：{"summary": "这里是图片描述内容"}。'
        )

    # tag
    if targets:
        pool_hint = (
            f"候选标签池：{_format_targets(targets)}\n"
            "优先从候选标签池中选择；如果候选池不足以准确描述图片，"
            "可以补充少量自创标签，但自创标签总数不超过 3 个。"
        )
    else:
        pool_hint = "请自由生成最能概括图片内容的标签。"
    return (
        "你是一个专业的图像打标助手。请为这张图片生成标签。\n"
        f"{pool_hint}\n"
        "规则：\n"
        "1. 输出 5~10 个标签；候选池不足 5 个时输出全部候选；\n"
        "2. 按置信度从高到低排列，最多 10 个，超过 10 个的会被自动截断；\n"
        "3. 每个标签给出 label 和 confidence（0~1 之间的小数）；\n"
        "4. 标签应覆盖场景、主体、动作、风格、色调等维度，避免近义重复。\n"
        f"{NO_EXPLAIN}\n"
        '输出格式：{"tags": [{"label": "标签", "confidence": 0.93}]}。'
    )


__all__ = ["build_prompt"]

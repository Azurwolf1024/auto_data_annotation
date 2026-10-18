# -*- coding: utf-8 -*-
"""数据契约模块。

用 pydantic v2 定义五种标注任务的结构化输出模型。
所有坐标字段均为 float 且归一化到 0~1（左上角为原点，右下角为 (1,1)）。

这些模型有两个用途：
1. 作为 Ollama `format` 参数的 JSON Schema 来源（强制结构化输出）；
2. 作为后处理阶段（validator / exporters）解析结果的载体。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# box
# --------------------------------------------------------------------------- #
class BoundingBox(BaseModel):
    """单个检测框。坐标为归一化值，x 向右、y 向下。"""

    label: str = Field(..., description="目标类别标签，必须来自候选标签列表")
    confidence: float = Field(..., description="置信度，0~1 之间的小数")
    x_min: float = Field(..., description="框左上角 x 坐标，归一化到 0~1")
    y_min: float = Field(..., description="框左上角 y 坐标，归一化到 0~1")
    x_max: float = Field(..., description="框右下角 x 坐标，归一化到 0~1")
    y_max: float = Field(..., description="框右下角 y 坐标，归一化到 0~1")


class BoxAnnotation(BaseModel):
    """box 任务的顶层结构。"""

    objects: List[BoundingBox] = Field(
        default_factory=list, description="图中所有检测到的目标框，没有目标时为空数组"
    )


# --------------------------------------------------------------------------- #
# point
# --------------------------------------------------------------------------- #
class PointItem(BaseModel):
    """单个物体中心点（不做结构化关键点）。"""

    label: str = Field(..., description="目标类别标签，必须来自候选标签列表")
    confidence: float = Field(..., description="置信度，0~1 之间的小数")
    x: float = Field(..., description="物体中心点 x 坐标，归一化到 0~1")
    y: float = Field(..., description="物体中心点 y 坐标，归一化到 0~1")


class PointAnnotation(BaseModel):
    """point 任务的顶层结构。"""

    points: List[PointItem] = Field(
        default_factory=list, description="图中所有物体的中心点，没有目标时为空数组"
    )


# --------------------------------------------------------------------------- #
# ocr
# --------------------------------------------------------------------------- #
class OcrBlock(BaseModel):
    """一个文字块。"""

    text: str = Field(..., description="识别到的文字内容")
    type: str = Field(
        ...,
        description=(
            "文字块类型，从以下取值中选择：title / paragraph / table / "
            "handwriting / signage / caption / other"
        ),
    )
    x_min: float = Field(..., description="文字块左上角 x 坐标，归一化到 0~1")
    y_min: float = Field(..., description="文字块左上角 y 坐标，归一化到 0~1")
    x_max: float = Field(..., description="文字块右下角 x 坐标，归一化到 0~1")
    y_max: float = Field(..., description="文字块右下角 y 坐标，归一化到 0~1")


class OcrAnnotation(BaseModel):
    """ocr 任务的顶层结构。"""

    blocks: List[OcrBlock] = Field(
        default_factory=list, description="图中所有文字块，没有文字时为空数组"
    )


# --------------------------------------------------------------------------- #
# caption
# --------------------------------------------------------------------------- #
class CaptionAnnotation(BaseModel):
    """caption 任务的顶层结构。"""

    summary: str = Field(..., description="对图片内容的中文描述，一到三句话")


# --------------------------------------------------------------------------- #
# tag
# --------------------------------------------------------------------------- #
class TagItem(BaseModel):
    """单个标签。"""

    label: str = Field(..., description="标签名，优先从候选标签池中选择")
    confidence: float = Field(..., description="置信度，0~1 之间的小数")


class TagAnnotation(BaseModel):
    """tag 任务的顶层结构。"""

    tags: List[TagItem] = Field(
        default_factory=list, description="5~10 个标签，按置信度从高到低排列"
    )


# --------------------------------------------------------------------------- #
# 统一结果封装
# --------------------------------------------------------------------------- #
class AnnotationResult(BaseModel):
    """单张图片的标注结果（含状态信息）。

    result 字段保存清洗后的 dict（与对应 *Annotation 模型结构一致）。
    """

    task_type: str = Field(..., description="标注任务类型：box/point/ocr/caption/tag")
    image_path: str = Field(..., description="图片路径")
    result: Dict = Field(default_factory=dict, description="清洗后的标注数据")
    status: str = Field("ok", description="处理状态：ok / failed")
    error: Optional[str] = Field(None, description="失败原因，成功时为 None")


# --------------------------------------------------------------------------- #
# 任务类型 -> schema 类 映射
# --------------------------------------------------------------------------- #
SCHEMA_MAP = {
    "box": BoxAnnotation,
    "point": PointAnnotation,
    "ocr": OcrAnnotation,
    "caption": CaptionAnnotation,
    "tag": TagAnnotation,
}

__all__ = [
    "BoundingBox",
    "BoxAnnotation",
    "PointItem",
    "PointAnnotation",
    "OcrBlock",
    "OcrAnnotation",
    "CaptionAnnotation",
    "TagItem",
    "TagAnnotation",
    "AnnotationResult",
    "SCHEMA_MAP",
]

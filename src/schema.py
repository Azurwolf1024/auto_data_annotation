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


class ZoomRegion(BaseModel):
    """模型主动请求「放大细看」的归一化区域。

    由 box / point 任务的结构化输出携带：模型在初次检测后，若认为某区域太小、
    太模糊或不确定，可声明一个归一化矩形（及简短原因）。harness 随后调用裁剪
    放大工具对该区域重新标注，并把结果映射回原图。这是「模型主动 call 工具」
    的实现方式（在结构化输出中表达放大意图）。
    """

    x_min: float = Field(..., description="区域左上角 x 坐标，归一化到 0~1")
    y_min: float = Field(..., description="区域左上角 y 坐标，归一化到 0~1")
    x_max: float = Field(..., description="区域右下角 x 坐标，归一化到 0~1")
    y_max: float = Field(..., description="区域右下角 y 坐标，归一化到 0~1")
    reason: str = Field("", description="为何要放大（可选，仅用于日志排查）")


class BoxAnnotation(BaseModel):
    """box 任务的顶层结构。"""

    objects: List[BoundingBox] = Field(
        default_factory=list, description="图中所有检测到的目标框，没有目标时为空数组"
    )
    zoom_regions: List[ZoomRegion] = Field(
        default_factory=list,
        description=(
            "模型认为需要放大细看的归一化区域；harness 会调用裁剪放大工具对每"
            "个区域重新标注并映射回原图，从而提升小目标/模糊目标的定位精度。"
            "无需放大时留空数组"
        ),
    )


# --------------------------------------------------------------------------- #
# verify（harness loop 二次验证）
# --------------------------------------------------------------------------- #
class VerifyAnnotation(BaseModel):
    """验证阶段：对单张「裁切子图」判断其中是否真有完整、可见的目标本体。

    由 harness 在 box / point 检测之后调用：把每个候选框（或点周围区域）裁切出来，
    单独喂给模型复核。

    - present=false：裁切内没有真实目标本体（如只有 HUD / 光效 / 文字横幅 /
      武器视图 / 击杀飘字等），调用方据此剔除该候选；
    - present=true：要求模型给出紧贴该真实目标的「相对本张裁切图」归一化修正框。
    """

    present: bool = Field(..., description="裁切区内是否真有完整、可见的目标本体")
    label: str = Field(
        "",
        description="目标类别，必须来自候选列表；无目标时为空串",
    )
    confidence: float = Field(
        0.0, description="对本次判断的置信度，0~1 之间的小数"
    )
    x_min: float = Field(0.0, description="修正框左上角 x，相对本张裁切图归一化到 0~1")
    y_min: float = Field(0.0, description="修正框左上角 y，相对本张裁切图归一化到 0~1")
    x_max: float = Field(0.0, description="修正框右下角 x，相对本张裁切图归一化到 0~1")
    y_max: float = Field(0.0, description="修正框右下角 y，相对本张裁切图归一化到 0~1")


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
    zoom_regions: List[ZoomRegion] = Field(
        default_factory=list,
        description=(
            "模型认为需要放大细看的归一化区域（围绕不确定的点取一小块即可）；"
            "harness 会调用裁剪放大工具对该区域重新标注并映射回原图。"
            "无需放大时留空数组"
        ),
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
    "ZoomRegion",
    "PointItem",
    "PointAnnotation",
    "OcrBlock",
    "OcrAnnotation",
    "CaptionAnnotation",
    "TagItem",
    "TagAnnotation",
    "AnnotationResult",
    "SCHEMA_MAP",
    "VerifyAnnotation",
]

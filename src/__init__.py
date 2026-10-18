# -*- coding: utf-8 -*-
"""auto-annotator：配置驱动的通用图像标注引擎（核心可 import 模块）。"""

from .config import DEFAULT_MODEL, OUTPUT_FORMATS, TASK_TYPES, Config
from .schema import SCHEMA_MAP, AnnotationResult
from .pipeline import run
from .exporters import export

__version__ = "0.1.0"

__all__ = [
    "Config",
    "TASK_TYPES",
    "OUTPUT_FORMATS",
    "DEFAULT_MODEL",
    "SCHEMA_MAP",
    "AnnotationResult",
    "run",
    "export",
    "__version__",
]

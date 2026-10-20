# -*- coding: utf-8 -*-
"""一键运行全部测试：MCP 协议层 + harness 集成层。

用法（在项目根目录执行）：
    python -m tests
    python tests/__main__.py

两种调用方式都会被正确解析：把 tests/ 目录与项目根目录都加入 sys.path，
因此既能 `import src...`（测试内部依赖），也能 `import test_*.py`（同目录用例）。
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    import test_mcp_zoom as t_mcp
    import test_harness_zoom as t_harness

    code = 0
    for suite in (t_mcp, t_harness):
        try:
            rc = suite.main()
        except Exception as exc:  # noqa: BLE001 - 某个套件崩溃不应吞掉整体退出码
            print(f"\n{suite.__name__} 运行异常：{type(exc).__name__}: {exc}")
            rc = 1
        code = code or rc
    return code


if __name__ == "__main__":
    sys.exit(main())

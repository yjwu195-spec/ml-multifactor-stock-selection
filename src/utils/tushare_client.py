from __future__ import annotations

import os
import time
from typing import Any

from .common import print_step


def get_pro_client(config: dict):
    try:
        import tushare as ts
    except ImportError as exc:
        raise ImportError(
            "未安装 tushare。请先执行 `pip install -r requirements.txt`。"
        ) from exc

    from dotenv import load_dotenv
    from .common import project_root

    load_dotenv(project_root() / ".env", override=False)
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise ValueError(
            "Missing TUSHARE_TOKEN. Set the environment variable or a local .env file."
        )

    return ts.pro_api(token)


def pro_call(pro: Any, api_name: str, config: dict, **kwargs):
    pause_seconds = config["tushare"]["request_pause_seconds"]
    retry_times = config["tushare"]["retry_times"]
    api = getattr(pro, api_name)
    last_error = None
    for attempt in range(1, retry_times + 1):
        try:
            df = api(**kwargs)
            time.sleep(pause_seconds)
            return df
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print_step(f"{api_name} 第 {attempt} 次请求失败: {exc}")
            time.sleep(max(1.0, pause_seconds * attempt * 3))
    raise RuntimeError(f"{api_name} 调用失败，已重试 {retry_times} 次。") from last_error

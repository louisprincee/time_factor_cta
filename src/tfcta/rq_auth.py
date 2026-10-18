"""从环境变量或 config/rqdata.env 读取米筐凭证。"""
from __future__ import annotations

import os
from pathlib import Path

from . import config as C

ENV_PATH = C.CONFIG_DIR / "rqdata.env"


def load_rqdata_env(path: Path | None = None) -> None:
    """用本地 env 文件填补尚未设置的米筐变量。"""
    path = Path(path) if path is not None else ENV_PATH
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


def rqdata_credentials(path: Path | None = None) -> tuple[str, ...] | None:
    """返回传给 rqdatac.init 的参数。"""
    load_rqdata_env(path)
    license_key = os.environ.get("RQDATAC_LICENSE", "").strip()
    if license_key:
        return (license_key,)
    username = os.environ.get("RQDATAC_USERNAME", "").strip()
    password = os.environ.get("RQDATAC_PASSWORD", "").strip()
    if username and password:
        return (username, password)
    return None

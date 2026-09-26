"""Load RiceQuant credentials from the environment or config/rqdata.env."""
from __future__ import annotations

import os
from pathlib import Path

from . import config as C

ENV_PATH = C.CONFIG_DIR / "rqdata.env"


def load_rqdata_env(path: Path | None = None) -> None:
    """Fill unset RQData variables from a local env file.

    Existing environment variables win, so a one-off export still overrides
    the file. Blank values and comments are ignored.
    """
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
    """Return positional arguments for ``rqdatac.init``.

    A license key is preferred. Username and password are the fallback.
    """
    load_rqdata_env(path)
    license_key = os.environ.get("RQDATAC_LICENSE", "").strip()
    if license_key:
        return (license_key,)
    username = os.environ.get("RQDATAC_USERNAME", "").strip()
    password = os.environ.get("RQDATAC_PASSWORD", "").strip()
    if username and password:
        return (username, password)
    return None

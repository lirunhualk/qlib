from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def _read_windows_env(name: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:
        return None

    env_keys = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    )
    for root, path in env_keys:
        try:
            with winreg.OpenKey(root, path) as key:
                value, _ = winreg.QueryValueEx(key, name)
                return value
        except OSError:
            continue
    return None


def _resolve_dsn(dsn: str | None = None) -> str:
    if dsn:
        return dsn
    for env_name in ("ASHARE_DB_URL", "ASHARE_MYSQL_DSN"):
        value = os.getenv(env_name) or _read_windows_env(env_name)
        if value:
            return value
    raise RuntimeError("ASHARE_DB_URL is not configured; pass --dsn or set the environment variable.")


def get_engine(dsn: str | None = None, **engine_kwargs) -> Engine:
    return create_engine(_resolve_dsn(dsn), **engine_kwargs)


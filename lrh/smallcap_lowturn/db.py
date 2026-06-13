from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


ASHARE_DB_URL_ENV = "ASHARE_DB_URL"
ASHARE_DB_HELPER_ENV = "ASHARE_DB_HELPER_DIR"
DEFAULT_ASHARE_DB_HELPER_DIR = Path(r"D:\github\aShare\code\db")


def _read_windows_env(name: str) -> str | None:
    if os.name != "nt":
        return None
    import winreg

    env_keys = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    )
    for root, path in env_keys:
        try:
            with winreg.OpenKey(root, path) as key:
                return winreg.QueryValueEx(key, name)[0]
        except OSError:
            continue
    return None


def get_engine(**engine_kwargs) -> Engine:
    """Return an aShare SQLAlchemy engine without storing credentials here."""

    helper_dir = Path(os.getenv(ASHARE_DB_HELPER_ENV, DEFAULT_ASHARE_DB_HELPER_DIR))
    if helper_dir.exists():
        helper_path = str(helper_dir)
        if helper_path not in sys.path:
            sys.path.append(helper_path)
        try:
            import db_config

            return db_config.get_engine(**engine_kwargs)
        except RuntimeError:
            pass

    db_url = os.getenv(ASHARE_DB_URL_ENV) or _read_windows_env(ASHARE_DB_URL_ENV)
    if not db_url:
        raise RuntimeError(
            f"{ASHARE_DB_URL_ENV} is not configured. Set it or provide "
            f"{ASHARE_DB_HELPER_ENV} pointing to aShare/code/db."
        )
    return create_engine(db_url, **engine_kwargs)


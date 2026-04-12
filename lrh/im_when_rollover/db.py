from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .config import DEFAULT_MYSQL_DSN


def get_engine(dsn: str | None = None) -> Engine:
    target_dsn = dsn or os.getenv("ASHARE_MYSQL_DSN") or DEFAULT_MYSQL_DSN
    return create_engine(target_dsn)

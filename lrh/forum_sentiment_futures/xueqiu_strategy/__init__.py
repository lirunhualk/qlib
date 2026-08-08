"""Authorized Xueqiu sentiment data adapters and intraday strategy."""

from .audit import BacktestDataAudit, audit_backtest_data
from .features import build_xueqiu_intraday_features
from .ingest import (
    first_observed_post_snapshot,
    load_xueqiu_export,
    normalize_xueqiu_frame,
)
from .signals import XueqiuSignalConfig, apply_xueqiu_signal

__all__ = [
    "BacktestDataAudit",
    "XueqiuSignalConfig",
    "apply_xueqiu_signal",
    "audit_backtest_data",
    "build_xueqiu_intraday_features",
    "first_observed_post_snapshot",
    "load_xueqiu_export",
    "normalize_xueqiu_frame",
]

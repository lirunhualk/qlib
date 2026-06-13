from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for the small-cap low-turnover strategy."""

    # 2020 data is used only for rolling features before the 5-year test window.
    start_date: int = 20200601
    test_start_date: int = 20210608
    end_date: int | None = None

    benchmark_code: str = "000852.SH"  # CSI 1000, closer to a small-cap universe.
    holdings_count: int = 200
    rebalance_interval: int = 20
    cost_bps: float = 10.0

    min_amt20: float = 20_000.0  # Tushare ts_daily.amount unit: thousand RMB.
    min_age_days: int = 365
    min_price: float = 3.0
    excluded_markets: tuple[str, ...] = ("北交所",)

    min_ret20: float = -0.08
    min_ret60: float = -0.20
    max_vol60: float = 1.20

    size_score_weight: float = 1.00
    turn_score_weight: float = 0.45
    vol_score_weight: float = 0.25
    ret20_score_weight: float = -0.15

    buy_limit_epsilon: float = 0.001
    sell_limit_epsilon: float = 0.001


DEFAULT_CONFIG = StrategyConfig()


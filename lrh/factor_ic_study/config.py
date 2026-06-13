from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorStudyConfig:
    start_date: int = 20200601
    test_start_date: int = 20210608
    end_date: int | None = None

    holdings_count: int = 200
    rebalance_interval: int = 20
    cost_bps: float = 10.0
    feature_set: str = "core"
    min_ic_obs: int = 500
    min_strategy_obs: int = 50

    min_amt20: float = 20_000.0
    min_age_days: int = 365
    min_price: float = 3.0
    excluded_markets: tuple[str, ...] = ("北交所",)


DEFAULT_CONFIG = FactorStudyConfig()

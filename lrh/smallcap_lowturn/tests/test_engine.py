from __future__ import annotations

import pandas as pd

from lrh.smallcap_lowturn.config import StrategyConfig
from lrh.smallcap_lowturn.engine import apply_trade_constraints, run_backtest


def test_apply_trade_constraints_blocks_limit_up_buy() -> None:
    config = StrategyConfig()
    desired = {"000001.SZ": 1.0}
    current: dict[str, float] = {}
    execution = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "open": [11.0],
            "up_limit": [11.0],
            "down_limit": [9.0],
        }
    )

    weights, blocked_buys, blocked_sells = apply_trade_constraints(desired, current, execution, config)

    assert weights == {}
    assert blocked_buys == 1
    assert blocked_sells == 0


def test_apply_trade_constraints_blocks_limit_down_sell() -> None:
    config = StrategyConfig()
    desired: dict[str, float] = {}
    current = {"000001.SZ": 1.0}
    execution = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "open": [9.0],
            "up_limit": [11.0],
            "down_limit": [9.0],
        }
    )

    weights, blocked_buys, blocked_sells = apply_trade_constraints(desired, current, execution, config)

    assert weights == {"000001.SZ": 1.0}
    assert blocked_buys == 0
    assert blocked_sells == 1


def test_run_backtest_smoke() -> None:
    config = StrategyConfig(
        start_date=20200101,
        test_start_date=20200106,
        holdings_count=1,
        rebalance_interval=1,
        min_amt20=0,
        min_age_days=0,
        min_price=0,
        min_ret20=-1,
        min_ret60=-1,
        max_vol60=10,
    )
    rows = []
    dates = [int(dt.strftime("%Y%m%d")) for dt in pd.bdate_range("2020-01-02", periods=80)]
    for idx, dt in enumerate(dates):
        rows.append(
            {
                "ts_code": "000001.SZ",
                "dt": dt,
                "open": 10.0 + idx * 0.1,
                "close": 10.0 + idx * 0.1,
                "pre_close": 10.0 + max(idx - 1, 0) * 0.1,
                "amount": 100000.0,
                "adj_factor": 1.0,
                "total_mv": 100000.0,
                "circ_mv": 90000.0,
                "turnover_rate_f": 1.0,
                "turnover_rate": 1.0,
                "up_limit": 100.0,
                "down_limit": 1.0,
                "name": "平安银行",
                "market": "主板",
                "list_status": "L",
                "list_date": 19910403,
            }
        )
    raw = pd.DataFrame(rows)
    benchmark = pd.DataFrame({"dt": dates, "bench": [0.0] * len(dates)})

    result = run_backtest(raw, benchmark, config)

    assert not result.daily.empty
    assert result.summary["days"] == len(result.daily)

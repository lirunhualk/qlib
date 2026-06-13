from __future__ import annotations

import pandas as pd

from lrh.factor_ic_study.config import FactorStudyConfig
from lrh.factor_ic_study.evaluation import compute_daily_ic, evaluate


def _sample_panel() -> pd.DataFrame:
    rows = []
    dates = [int(dt.strftime("%Y%m%d")) for dt in pd.bdate_range("2020-01-02", periods=35)]
    for dt_idx, dt in enumerate(dates):
        for code_idx in range(20):
            rows.append(
                {
                    "dt": dt,
                    "ts_code": f"{code_idx:06d}.SZ",
                    "name": f"s{code_idx}",
                    "base_ok": True,
                    "exec_ret_fwd": code_idx / 1000.0 + dt_idx / 100000.0,
                    "good_factor": float(code_idx),
                    "bad_factor": float(-code_idx),
                }
            )
    return pd.DataFrame(rows)


def test_compute_daily_ic_detects_monotonic_factor() -> None:
    config = FactorStudyConfig(test_start_date=20200102, min_ic_obs=10)
    daily_ic = compute_daily_ic(_sample_panel(), ["good_factor", "bad_factor"], config)

    assert daily_ic["good_factor"].mean() > 0.99
    assert daily_ic["bad_factor"].mean() < -0.99


def test_evaluate_smoke() -> None:
    config = FactorStudyConfig(test_start_date=20200102, min_ic_obs=10, min_strategy_obs=5, holdings_count=5, rebalance_interval=5)
    result = evaluate(_sample_panel(), ["good_factor", "bad_factor"], config)

    assert not result.factor_metrics.empty
    assert not result.correlation_summary.empty


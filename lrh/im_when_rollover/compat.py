"""Small compatibility helpers that mirror qlib behavior when local qlib isn't importable."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def qlib_style_risk_analysis(
    returns: pd.Series,
    *,
    N: int | None = None,
    freq: str = "day",
    mode: Literal["sum", "product"] = "product",
) -> pd.DataFrame:
    """Mirror qlib.contrib.evaluate.risk_analysis for this local research package."""

    if N is None:
        freq_scaler = {
            "min": 240 * 238,
            "minute": 240 * 238,
            "day": 252,
            "week": 50,
            "month": 12,
        }
        N = freq_scaler[freq]

    clean = pd.Series(returns, dtype="float64").dropna()
    if clean.empty:
        data = {
            "mean": np.nan,
            "std": np.nan,
            "annualized_return": np.nan,
            "information_ratio": np.nan,
            "max_drawdown": np.nan,
        }
        return pd.Series(data).to_frame("risk")

    if mode == "sum":
        mean = clean.mean()
        std = clean.std(ddof=1)
        annualized_return = mean * N
        max_drawdown = (clean.cumsum() - clean.cumsum().cummax()).min()
    elif mode == "product":
        cumulative_curve = (1 + clean).cumprod()
        mean = cumulative_curve.iloc[-1] ** (1 / len(clean)) - 1
        std = np.log1p(clean).std(ddof=1)
        cumulative_return = cumulative_curve.iloc[-1] - 1
        annualized_return = (1 + cumulative_return) ** (N / len(clean)) - 1
        max_drawdown = (cumulative_curve / cumulative_curve.cummax() - 1).min()
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    information_ratio = np.nan
    if pd.notna(std) and std != 0:
        information_ratio = mean / std * np.sqrt(N)

    data = {
        "mean": mean,
        "std": std,
        "annualized_return": annualized_return,
        "information_ratio": information_ratio,
        "max_drawdown": max_drawdown,
    }
    return pd.Series(data).to_frame("risk")

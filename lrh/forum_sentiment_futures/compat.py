"""Small compatibility helpers that mirror qlib daily risk metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def qlib_style_risk_analysis(
    returns: pd.Series,
    *,
    N: int = 252,
    mode: Literal["sum", "product"] = "product",
) -> pd.DataFrame:
    clean = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return pd.Series(
            {
                "mean": np.nan,
                "std": np.nan,
                "annualized_return": np.nan,
                "information_ratio": np.nan,
                "max_drawdown": np.nan,
            }
        ).to_frame("risk")

    if mode == "sum":
        mean = clean.mean()
        std = clean.std(ddof=1)
        annualized_return = mean * N
        drawdown = clean.cumsum() - clean.cumsum().cummax()
        max_drawdown = drawdown.min()
    elif mode == "product":
        curve = (1.0 + clean).cumprod()
        mean = curve.iloc[-1] ** (1.0 / len(clean)) - 1.0
        std = np.log1p(clean).std(ddof=1)
        total_return = curve.iloc[-1] - 1.0
        annualized_return = (1.0 + total_return) ** (N / len(clean)) - 1.0
        max_drawdown = (curve / curve.cummax() - 1.0).min()
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    information_ratio = np.nan
    if pd.notna(std) and std != 0:
        information_ratio = mean / std * np.sqrt(N)

    return pd.Series(
        {
            "mean": mean,
            "std": std,
            "annualized_return": annualized_return,
            "information_ratio": information_ratio,
            "max_drawdown": max_drawdown,
        }
    ).to_frame("risk")


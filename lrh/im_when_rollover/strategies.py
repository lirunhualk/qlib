from __future__ import annotations

from typing import Callable

import pandas as pd

from .config import StrategySpec


def select_target_contract(
    spec: StrategySpec,
    snapshot: pd.DataFrame,
    current_contract: str | None,
) -> str:
    selector_map: dict[str, Callable[[StrategySpec, pd.DataFrame, str | None], str]] = {
        "dominant_volume": _dominant_volume,
        "dominant_oi": _dominant_oi,
        "max_carry_daily": _max_carry_daily,
        "max_carry_daily_liquid": _max_carry_daily_liquid,
        "roll_chain": _roll_chain,
        "best_carry_hold": _best_carry_hold,
    }
    return selector_map[spec.selector](spec, snapshot.copy(), current_contract)


def _eligible(snapshot: pd.DataFrame) -> pd.DataFrame:
    eligible = snapshot[snapshot["trade_days_to_expiry"] > 0].copy()
    if eligible.empty:
        raise ValueError("No eligible contracts with at least one trading day remaining")
    return eligible


def _sorted_by_expiry(snapshot: pd.DataFrame) -> pd.DataFrame:
    return snapshot.sort_values(["delist_date", "trade_days_to_expiry"]).sort_index()


def _pick_initial_by_rank(snapshot: pd.DataFrame, rank: int) -> str:
    ordered = _sorted_by_expiry(_eligible(snapshot))
    index = min(rank - 1, len(ordered) - 1)
    return str(ordered.iloc[index]["ts_code"])


def _pick_next_contract(snapshot: pd.DataFrame, current_contract: str | None) -> str:
    ordered = _sorted_by_expiry(_eligible(snapshot))
    if not current_contract or current_contract not in snapshot.index:
        return str(ordered.iloc[0]["ts_code"])

    current_delist = snapshot.loc[current_contract, "delist_date"]
    later = ordered[ordered["delist_date"] > current_delist]
    if not later.empty:
        return str(later.iloc[0]["ts_code"])

    fallback = ordered[ordered["ts_code"] != current_contract]
    if fallback.empty:
        return str(ordered.iloc[-1]["ts_code"])
    return str(fallback.iloc[-1]["ts_code"])


def _dominant_metric(snapshot: pd.DataFrame, current_contract: str | None, metric: str) -> str:
    ranked = _eligible(snapshot).copy()
    ranked["prefer_current"] = ranked["ts_code"].eq(current_contract)
    ranked = ranked.sort_values([metric, "prefer_current", "delist_date"], ascending=[False, False, True])
    return str(ranked.iloc[0]["ts_code"])


def _dominant_volume(spec: StrategySpec, snapshot: pd.DataFrame, current_contract: str | None) -> str:
    return _dominant_metric(snapshot, current_contract, "vol")


def _dominant_oi(spec: StrategySpec, snapshot: pd.DataFrame, current_contract: str | None) -> str:
    return _dominant_metric(snapshot, current_contract, "oi")


def _max_carry(snapshot: pd.DataFrame) -> str:
    ranked = _eligible(snapshot).copy()
    ranked = ranked.sort_values(
        ["annualized_carry", "delist_date", "amount"],
        ascending=[False, True, False],
        na_position="last",
    )
    return str(ranked.iloc[0]["ts_code"])


def _max_carry_daily(spec: StrategySpec, snapshot: pd.DataFrame, current_contract: str | None) -> str:
    return _max_carry(snapshot)


def _max_carry_daily_liquid(spec: StrategySpec, snapshot: pd.DataFrame, current_contract: str | None) -> str:
    eligible = _eligible(snapshot).copy()
    top_n = spec.liquid_top_n or 2
    liquid = eligible.sort_values(["amount", "oi", "delist_date"], ascending=[False, False, True]).head(top_n)
    if liquid.empty:
        liquid = eligible
    return _max_carry(liquid)


def _roll_chain(spec: StrategySpec, snapshot: pd.DataFrame, current_contract: str | None) -> str:
    if current_contract is None or current_contract not in snapshot.index:
        return _pick_initial_by_rank(snapshot, spec.initial_rank or 1)

    current_row = snapshot.loc[current_contract]
    if int(current_row["trade_days_to_expiry"]) <= int(spec.roll_days_before_expiry or 0):
        return _pick_next_contract(snapshot, current_contract)

    if int(current_row["trade_days_to_expiry"]) <= 0:
        return _pick_next_contract(snapshot, current_contract)

    return current_contract


def _best_carry_hold(spec: StrategySpec, snapshot: pd.DataFrame, current_contract: str | None) -> str:
    if current_contract is None or current_contract not in snapshot.index:
        return _max_carry(snapshot)

    current_row = snapshot.loc[current_contract]
    if int(current_row["trade_days_to_expiry"]) <= 0:
        eligible = _eligible(snapshot)
        later = eligible[eligible["delist_date"] > current_row["delist_date"]]
        return _max_carry(later if not later.empty else eligible)

    return current_contract

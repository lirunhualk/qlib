from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from ..config import FUTURE_SPECS
from .ingest import split_symbol_codes


DIRECT_INDEX_SYMBOLS = {
    "SH000300": "IF",
    "SH000905": "IC",
    "SH000016": "IH",
    "SH000852": "IM",
}


@dataclass(frozen=True)
class MembershipCoverage:
    fut_code: str
    first_snapshot: pd.Timestamp | None
    last_snapshot: pd.Timestamp | None
    snapshot_count: int
    mapped_posts: int
    direct_index_posts: int
    max_staleness_days: float


def _tushare_to_xueqiu_symbol(value: str) -> str:
    code, market = str(value).split(".", maxsplit=1)
    return f"{market.upper()}{code}"


def load_index_membership(
    engine: Engine,
    *,
    fut_codes: list[str] | tuple[str, ...],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    index_codes = [FUTURE_SPECS[code].spot_code for code in fut_codes]
    query = (
        text(
            """
            SELECT index_code, con_code, weight, dt
            FROM ts_index_weight
            WHERE index_code IN :index_codes
              AND dt BETWEEN :start_dt AND :end_dt
            """
        )
        .bindparams(bindparam("index_codes", expanding=True))
    )
    start_dt = int(
        (pd.Timestamp(start_date).normalize() - pd.Timedelta(days=400)).strftime(
            "%Y%m%d"
        )
    )
    end_dt = int(pd.Timestamp(end_date).normalize().strftime("%Y%m%d"))
    with engine.connect() as connection:
        frame = pd.read_sql_query(
            query,
            connection,
            params={
                "index_codes": index_codes,
                "start_dt": start_dt,
                "end_dt": end_dt,
            },
        )
    if frame.empty:
        raise ValueError("ts_index_weight returned no constituent snapshots")
    frame["snapshot_date"] = pd.to_datetime(
        frame["dt"].astype(str), format="%Y%m%d", errors="coerce"
    )
    frame["symbol"] = frame["con_code"].map(_tushare_to_xueqiu_symbol)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    spot_to_future = {
        FUTURE_SPECS[code].spot_code: code for code in fut_codes
    }
    frame["fut_code"] = frame["index_code"].map(spot_to_future)
    return frame.dropna(
        subset=["snapshot_date", "symbol", "fut_code"]
    ).reset_index(drop=True)


def _map_one_future(
    posts: pd.DataFrame,
    exploded: pd.DataFrame,
    membership: pd.DataFrame,
    fut_code: str,
) -> tuple[pd.DataFrame, MembershipCoverage]:
    direct_symbols = {
        symbol for symbol, code in DIRECT_INDEX_SYMBOLS.items() if code == fut_code
    }
    direct_ids = set(
        exploded.loc[exploded["symbol"].isin(direct_symbols), "_post_row_id"].tolist()
    )

    weights = membership[membership["fut_code"] == fut_code].copy()
    snapshot_dates = (
        weights["snapshot_date"].drop_duplicates().sort_values().to_numpy(
            dtype="datetime64[ns]"
        )
    )
    candidate = exploded[~exploded["symbol"].isin(DIRECT_INDEX_SYMBOLS)].copy()
    if len(snapshot_dates) and not candidate.empty:
        post_dates = candidate["created_at"].dt.normalize().to_numpy(
            dtype="datetime64[ns]"
        )
        locations = np.searchsorted(snapshot_dates, post_dates, side="right") - 1
        valid = locations >= 0
        candidate = candidate.loc[valid].copy()
        candidate["membership_snapshot_date"] = pd.to_datetime(
            snapshot_dates[locations[valid]]
        )
        candidate = candidate.merge(
            weights[["snapshot_date", "symbol", "weight"]],
            left_on=["membership_snapshot_date", "symbol"],
            right_on=["snapshot_date", "symbol"],
            how="inner",
            validate="many_to_one",
        )
        candidate["membership_staleness_days"] = (
            candidate["created_at"].dt.normalize()
            - candidate["membership_snapshot_date"]
        ).dt.days
    else:
        candidate = pd.DataFrame(
            columns=[
                "_post_row_id",
                "weight",
                "membership_snapshot_date",
                "membership_staleness_days",
            ]
        )

    membership_summary = (
        candidate.groupby("_post_row_id", as_index=False)
        .agg(
            constituent_weight=("weight", "max"),
            membership_snapshot_date=("membership_snapshot_date", "max"),
            membership_staleness_days=("membership_staleness_days", "max"),
        )
        if not candidate.empty
        else pd.DataFrame(
            columns=[
                "_post_row_id",
                "constituent_weight",
                "membership_snapshot_date",
                "membership_staleness_days",
            ]
        )
    )
    matched_ids = set(membership_summary["_post_row_id"].tolist()) | direct_ids
    mapped = posts[posts["_post_row_id"].isin(matched_ids)].copy()
    mapped = mapped.merge(
        membership_summary,
        on="_post_row_id",
        how="left",
        validate="one_to_one",
    )
    direct_mask = mapped["_post_row_id"].isin(direct_ids)
    mapped["direct_index_mention"] = direct_mask
    mapped.loc[direct_mask, "constituent_weight"] = 100.0
    mapped.loc[direct_mask, "membership_staleness_days"] = 0.0
    mapped["fut_code"] = fut_code
    mapped["board_code"] = f"xueqiu_{fut_code}"
    mapped["post_id"] = (
        mapped["provider"].astype(str)
        + ":"
        + mapped["post_id"].astype(str)
    )
    mapped["publish_time"] = mapped["created_at"]
    mapped["title"] = mapped["text"].where(
        mapped["is_content_point_in_time"], ""
    )

    first_snapshot = (
        pd.Timestamp(snapshot_dates[0]) if len(snapshot_dates) else None
    )
    last_snapshot = (
        pd.Timestamp(snapshot_dates[-1]) if len(snapshot_dates) else None
    )
    staleness = pd.to_numeric(
        mapped["membership_staleness_days"], errors="coerce"
    )
    coverage = MembershipCoverage(
        fut_code=fut_code,
        first_snapshot=first_snapshot,
        last_snapshot=last_snapshot,
        snapshot_count=len(snapshot_dates),
        mapped_posts=len(mapped),
        direct_index_posts=int(direct_mask.sum()),
        max_staleness_days=float(staleness.max()) if staleness.notna().any() else 0.0,
    )
    return mapped, coverage


def assign_posts_to_futures(
    posts: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    fut_codes: list[str] | tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = posts.copy().reset_index(drop=True)
    source["_post_row_id"] = np.arange(len(source), dtype=np.int64)
    exploded = source[["_post_row_id", "created_at", "symbol_codes"]].copy()
    exploded["symbol"] = exploded["symbol_codes"].map(split_symbol_codes)
    exploded = exploded.explode("symbol")
    exploded = exploded[exploded["symbol"].notna() & exploded["symbol"].ne("")]

    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    for fut_code in fut_codes:
        mapped, coverage = _map_one_future(
            source, exploded, membership, fut_code
        )
        if not mapped.empty:
            frames.append(mapped)
        coverage_rows.append(coverage.__dict__)
    columns_to_drop = ["_post_row_id"]
    result = (
        pd.concat(frames, ignore_index=True).drop(columns=columns_to_drop)
        if frames
        else pd.DataFrame()
    )
    if not result.empty:
        result = result.sort_values(
            ["fut_code", "created_at", "post_id"]
        ).reset_index(drop=True)
    return result, pd.DataFrame(coverage_rows)

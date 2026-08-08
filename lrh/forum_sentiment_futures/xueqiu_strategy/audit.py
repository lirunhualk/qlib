from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestDataAudit:
    eligible: bool
    reasons: tuple[str, ...]
    requested_start: str
    requested_end: str
    row_count: int
    first_post: str | None
    last_post: str | None
    authorized_fraction: float
    point_in_time_content_fraction: float
    point_in_time_metrics_fraction: float
    precise_timestamp_fraction: float
    duplicate_fraction: float
    by_code: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "row_count": self.row_count,
            "first_post": self.first_post,
            "last_post": self.last_post,
            "authorized_fraction": self.authorized_fraction,
            "point_in_time_content_fraction": self.point_in_time_content_fraction,
            "point_in_time_metrics_fraction": self.point_in_time_metrics_fraction,
            "precise_timestamp_fraction": self.precise_timestamp_fraction,
            "duplicate_fraction": self.duplicate_fraction,
            "by_code": self.by_code.to_dict(orient="records"),
        }


def _fraction(values: pd.Series) -> float:
    return float(values.fillna(False).astype(bool).mean()) if len(values) else 0.0


def _format_timestamp(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def audit_backtest_data(
    posts: pd.DataFrame,
    trading_calendar: pd.DataFrame,
    *,
    fut_codes: list[str] | tuple[str, ...],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_trading_day_coverage: float = 0.80,
    min_point_in_time_content: float = 0.95,
    min_precise_timestamp_fraction: float = 0.99,
    min_median_posts_per_day: float = 5.0,
    max_membership_staleness_days: float = 180.0,
) -> BacktestDataAudit:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    reasons: list[str] = []
    source = posts.copy()
    if source.empty:
        reasons.append("没有可用于回测的雪球帖子")
        by_code = pd.DataFrame(
            {
                "fut_code": list(fut_codes),
                "post_count": 0,
                "first_post": None,
                "last_post": None,
                "trading_day_coverage": 0.0,
                "median_posts_per_active_day": 0.0,
                "point_in_time_content_fraction": 0.0,
                "max_membership_staleness_days": np.nan,
                "eligible": False,
            }
        )
        return BacktestDataAudit(
            eligible=False,
            reasons=tuple(reasons),
            requested_start=start.date().isoformat(),
            requested_end=end.date().isoformat(),
            row_count=0,
            first_post=None,
            last_post=None,
            authorized_fraction=0.0,
            point_in_time_content_fraction=0.0,
            point_in_time_metrics_fraction=0.0,
            precise_timestamp_fraction=0.0,
            duplicate_fraction=0.0,
            by_code=by_code,
        )

    source["created_at"] = pd.to_datetime(source["created_at"], errors="coerce")
    source = source[source["created_at"].notna()].copy()
    source["trade_date"] = source["created_at"].dt.normalize()
    source = source[source["trade_date"].between(start, end)]
    if source.empty:
        reasons.append("帖子时间与请求的回测区间没有交集")

    authorized_fraction = _fraction(source.get("authorized_use", pd.Series(False, index=source.index)))
    content_fraction = _fraction(
        source.get(
            "is_content_point_in_time",
            pd.Series(False, index=source.index),
        )
    )
    metrics_fraction = _fraction(
        source.get(
            "is_metrics_point_in_time",
            pd.Series(False, index=source.index),
        )
    )
    precise_timestamp_fraction = (
        float(source["created_at"].ne(source["created_at"].dt.normalize()).mean())
        if len(source)
        else 0.0
    )
    duplicate_mask = source.duplicated(
        subset=["fut_code", "post_id"], keep="first"
    ) if {"fut_code", "post_id"}.issubset(source) else pd.Series(False, index=source.index)
    duplicate_fraction = float(duplicate_mask.mean()) if len(duplicate_mask) else 0.0

    if authorized_fraction < 1.0:
        reasons.append("数据授权元数据未确认 authorized_use=true")
    if content_fraction < min_point_in_time_content:
        reasons.append(
            "帖子内容不是发帖时点快照，不能把事后编辑文本用于盘中回测"
        )
    if precise_timestamp_fraction < min_precise_timestamp_fraction:
        reasons.append("发布时间缺少盘中时分秒，不能映射到 15 分钟 K 线")
    if duplicate_fraction > 0.001:
        reasons.append("同一品种存在重复帖子，需先按帖子 ID 去重")

    first_post = source["created_at"].min() if not source.empty else None
    last_post = source["created_at"].max() if not source.empty else None
    if first_post is None or first_post.normalize() > start + pd.Timedelta(days=31):
        reasons.append("历史起点不足，未覆盖请求区间开头")
    if last_post is None or last_post.normalize() < end - pd.Timedelta(days=31):
        reasons.append("历史终点不足，未覆盖请求区间结尾")

    calendar = trading_calendar.copy()
    calendar["trade_date"] = pd.to_datetime(
        calendar["trade_date"], errors="coerce"
    ).dt.normalize()
    rows: list[dict] = []
    for fut_code in fut_codes:
        code_posts = source[source.get("fut_code", "").eq(fut_code)].copy()
        code_calendar = calendar[
            calendar["fut_code"].eq(fut_code)
            & calendar["trade_date"].between(start, end)
        ]
        expected_dates = set(code_calendar["trade_date"].dropna().tolist())
        daily_counts = (
            code_posts.groupby("trade_date")["post_id"].nunique()
            if not code_posts.empty
            else pd.Series(dtype=float)
        )
        active_dates = set(daily_counts.index.tolist()) & expected_dates
        day_coverage = (
            len(active_dates) / len(expected_dates) if expected_dates else 0.0
        )
        median_posts = (
            float(daily_counts.loc[list(active_dates)].median())
            if active_dates
            else 0.0
        )
        pit_fraction = _fraction(
            code_posts.get(
                "is_content_point_in_time",
                pd.Series(False, index=code_posts.index),
            )
        )
        staleness = pd.to_numeric(
            code_posts.get(
                "membership_staleness_days",
                pd.Series(0.0, index=code_posts.index),
            ),
            errors="coerce",
        )
        max_staleness = (
            float(staleness.max()) if staleness.notna().any() else 0.0
        )
        code_eligible = (
            len(code_posts) > 0
            and day_coverage >= min_trading_day_coverage
            and median_posts >= min_median_posts_per_day
            and pit_fraction >= min_point_in_time_content
            and max_staleness <= max_membership_staleness_days
        )
        rows.append(
            {
                "fut_code": fut_code,
                "post_count": len(code_posts),
                "first_post": _format_timestamp(
                    code_posts["created_at"].min()
                    if not code_posts.empty
                    else None
                ),
                "last_post": _format_timestamp(
                    code_posts["created_at"].max()
                    if not code_posts.empty
                    else None
                ),
                "trading_day_coverage": day_coverage,
                "median_posts_per_active_day": median_posts,
                "point_in_time_content_fraction": pit_fraction,
                "max_membership_staleness_days": max_staleness,
                "eligible": code_eligible,
            }
        )
        if not code_eligible:
            reasons.append(
                f"{fut_code} 未通过覆盖率/帖子密度/点时内容/成分时效审计"
            )

    by_code = pd.DataFrame(rows)
    reasons = list(dict.fromkeys(reasons))
    return BacktestDataAudit(
        eligible=not reasons and bool(by_code["eligible"].all()),
        reasons=tuple(reasons),
        requested_start=start.date().isoformat(),
        requested_end=end.date().isoformat(),
        row_count=len(source),
        first_post=_format_timestamp(first_post),
        last_post=_format_timestamp(last_post),
        authorized_fraction=authorized_fraction,
        point_in_time_content_fraction=content_fraction,
        point_in_time_metrics_fraction=metrics_fraction,
        precise_timestamp_fraction=precise_timestamp_fraction,
        duplicate_fraction=duplicate_fraction,
        by_code=by_code,
    )


def write_audit_report(audit: BacktestDataAudit, output_dir: Path | str) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "data_audit.json").write_text(
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit.by_code.to_csv(
        destination / "data_audit_by_code.csv",
        index=False,
        encoding="utf-8-sig",
    )
    status = "通过" if audit.eligible else "阻断"
    reason_lines = (
        "\n".join(f"- {reason}" for reason in audit.reasons)
        if audit.reasons
        else "- 无"
    )
    table = audit.by_code.to_markdown(index=False) if not audit.by_code.empty else "无数据"
    report = f"""# 雪球盘中回测数据审计

状态：**{status}**

请求区间：`{audit.requested_start}` 至 `{audit.requested_end}`

帖子行数：`{audit.row_count}`

点时内容比例：`{audit.point_in_time_content_fraction:.2%}`

授权确认比例：`{audit.authorized_fraction:.2%}`

精确时间比例：`{audit.precise_timestamp_fraction:.2%}`

## 阻断原因

{reason_lines}

## 分品种覆盖

{table}

## 审计原则

- 默认不使用点赞、回复、转发和浏览量；除非每个数值都有对应历史快照时间。
- 历史导出中的当前文本可能包含事后编辑，只有供应商明确提供发帖时点快照才进入情绪特征。
- 数据不足时不生成收益率、年化、Sharpe 或“策略改进”结论。
"""
    (destination / "DATA_AUDIT.md").write_text(report, encoding="utf-8")

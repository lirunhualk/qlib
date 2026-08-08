from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = (
    "provider",
    "post_id",
    "user_id",
    "user_name",
    "created_at",
    "observed_at",
    "content_observed_at",
    "metrics_observed_at",
    "symbol_codes",
    "title",
    "text",
    "source",
    "source_url",
    "like_count",
    "reply_count",
    "repost_count",
    "view_count",
    "original_post_id",
    "is_content_point_in_time",
    "is_metrics_point_in_time",
    "authorized_use",
)

DATASEED_ALIASES = {
    "provider": ("provider", "供应商", "数据供应商"),
    "post_id": ("post_id", "status_id", "ID", "帖子ID"),
    "user_id": ("user_id", "发帖者ID", "帖子发表者ID"),
    "user_name": ("user_name", "screen_name", "帖子发表者呢称", "帖子发表者昵称"),
    "created_at": ("created_at", "publish_time", "发帖时间", "发布时间"),
    "observed_at": ("observed_at", "ingested_at", "采集时间", "首次采集时间"),
    "content_observed_at": (
        "content_observed_at",
        "content_snapshot_at",
        "内容快照时间",
    ),
    "metrics_observed_at": (
        "metrics_observed_at",
        "interaction_snapshot_at",
        "互动快照时间",
    ),
    "symbol_codes": ("symbol_codes", "symbols", "股票代码", "证券代码"),
    "title": ("title", "帖子标题", "标题"),
    "text": ("text", "content", "帖子内容", "正文"),
    "source": ("source", "来源"),
    "source_url": ("source_url", "url", "帖子链接", "原文链接"),
    "like_count": ("like_count", "点赞量", "点赞数"),
    "reply_count": ("reply_count", "评论量", "评论数", "回复量"),
    "repost_count": ("repost_count", "转发量", "转发数"),
    "view_count": ("view_count", "浏览量", "阅读量"),
    "original_post_id": (
        "original_post_id",
        "帖子转评原发表者ID",
        "原帖ID",
    ),
}

SYMBOL_PATTERN = re.compile(r"(?:(SH|SZ)(\d{6})|(\d{6})\.(SH|SZ))", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


class XueqiuDataError(ValueError):
    """Raised when an export cannot be normalized without guessing."""


@dataclass(frozen=True)
class AuthorizationMetadata:
    provider: str
    authorized_use: bool
    content_snapshot_at_creation: bool
    interaction_metrics_point_in_time: bool
    timezone: str
    license_reference: str

    @classmethod
    def from_path(cls, path: Path | str) -> "AuthorizationMetadata":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {
            "provider",
            "authorized_use",
            "content_snapshot_at_creation",
            "interaction_metrics_point_in_time",
            "timezone",
            "license_reference",
        }
        missing = required.difference(payload)
        if missing:
            raise XueqiuDataError(
                f"Authorization metadata missing fields: {sorted(missing)}"
            )
        if payload["timezone"] != "Asia/Shanghai":
            raise XueqiuDataError("Xueqiu timestamps must be declared as Asia/Shanghai")
        return cls(
            provider=str(payload["provider"]).strip(),
            authorized_use=bool(payload["authorized_use"]),
            content_snapshot_at_creation=bool(
                payload["content_snapshot_at_creation"]
            ),
            interaction_metrics_point_in_time=bool(
                payload["interaction_metrics_point_in_time"]
            ),
            timezone=str(payload["timezone"]),
            license_reference=str(payload["license_reference"]).strip(),
        )


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data)

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() != "img":
            return
        attributes = dict(attrs)
        marker = attributes.get("alt") or attributes.get("title")
        if marker:
            self.parts.append(marker)


def html_to_visible_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    parser = _VisibleTextParser()
    try:
        parser.feed(str(value))
        parser.close()
        text = " ".join(parser.parts)
    except Exception:
        text = str(value)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_symbol_codes(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    symbols: list[str] = []
    for match in SYMBOL_PATTERN.finditer(str(value)):
        if match.group(1):
            market, code = match.group(1).upper(), match.group(2)
        else:
            market, code = match.group(4).upper(), match.group(3)
        symbol = f"{market}{code}"
        if symbol not in symbols:
            symbols.append(symbol)
    return "|".join(symbols)


def split_symbol_codes(value: Any) -> list[str]:
    normalized = normalize_symbol_codes(value)
    return normalized.split("|") if normalized else []


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    direct = {str(column): str(column) for column in frame.columns}
    folded = {str(column).casefold(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias in direct:
            return direct[alias]
        if alias.casefold() in folded:
            return folded[alias.casefold()]
    return None


def _series_from_alias(
    frame: pd.DataFrame, canonical_name: str, default: Any = ""
) -> pd.Series:
    column = _find_column(frame, DATASEED_ALIASES[canonical_name])
    if column is None:
        return pd.Series(default, index=frame.index)
    return frame[column]


def _string_series(values: pd.Series) -> pd.Series:
    return values.fillna("").map(
        lambda value: str(int(value))
        if isinstance(value, (int, np.integer))
        or (isinstance(value, float) and np.isfinite(value) and value.is_integer())
        else str(value).strip()
    )


def _read_delimited(path: Path, separator: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, sep=separator, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise XueqiuDataError(f"Unable to decode {path}: {last_error}")


def _read_export(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_delimited(path, ",")
    if suffix == ".tsv":
        return _read_delimited(path, "\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "items", "records", "list"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise XueqiuDataError("JSON export must contain a list of post records")
        return pd.DataFrame(payload)
    raise XueqiuDataError(f"Unsupported Xueqiu export format: {suffix}")


def normalize_xueqiu_frame(
    frame: pd.DataFrame,
    *,
    metadata: AuthorizationMetadata | None = None,
    default_provider: str = "authorized_xueqiu_export",
    default_observed_at: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    normalized = pd.DataFrame(index=frame.index)
    for column in DATASEED_ALIASES:
        normalized[column] = _series_from_alias(frame, column)

    normalized["post_id"] = _string_series(normalized["post_id"])
    normalized["user_id"] = _string_series(normalized["user_id"])
    normalized["user_name"] = _string_series(normalized["user_name"])
    normalized["original_post_id"] = _string_series(
        normalized["original_post_id"]
    )
    normalized["created_at"] = pd.to_datetime(
        normalized["created_at"], errors="coerce"
    )
    normalized["observed_at"] = pd.to_datetime(
        normalized["observed_at"], errors="coerce"
    )
    normalized["content_observed_at"] = pd.to_datetime(
        normalized["content_observed_at"], errors="coerce"
    )
    normalized["metrics_observed_at"] = pd.to_datetime(
        normalized["metrics_observed_at"], errors="coerce"
    )
    if default_observed_at is not None:
        observed = pd.Timestamp(default_observed_at)
        normalized["observed_at"] = normalized["observed_at"].fillna(observed)

    provider = metadata.provider if metadata else default_provider
    supplied_provider = _string_series(normalized["provider"])
    normalized["provider"] = supplied_provider.mask(supplied_provider.eq(""), provider)
    normalized["symbol_codes"] = normalized["symbol_codes"].map(
        normalize_symbol_codes
    )
    normalized["title"] = normalized["title"].map(html_to_visible_text)
    normalized["text"] = normalized["text"].map(html_to_visible_text)
    normalized["text"] = (
        normalized["title"].str.cat(normalized["text"], sep=" ").str.strip()
    )
    normalized["source"] = _string_series(normalized["source"])
    normalized["source_url"] = _string_series(normalized["source_url"])

    for column in ("like_count", "reply_count", "repost_count", "view_count"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if metadata and metadata.content_snapshot_at_creation:
        normalized["content_observed_at"] = normalized[
            "content_observed_at"
        ].fillna(normalized["created_at"])
    normalized["is_content_point_in_time"] = (
        normalized["content_observed_at"].notna()
        & normalized["created_at"].notna()
        & (
            normalized["content_observed_at"]
            <= normalized["created_at"] + pd.Timedelta(minutes=15)
        )
    )
    if metadata and metadata.interaction_metrics_point_in_time:
        normalized["metrics_observed_at"] = normalized[
            "metrics_observed_at"
        ].fillna(normalized["observed_at"])
    normalized["is_metrics_point_in_time"] = normalized[
        "metrics_observed_at"
    ].notna()
    normalized["authorized_use"] = bool(metadata and metadata.authorized_use)

    normalized = normalized.dropna(subset=["created_at"])
    normalized = normalized[
        normalized["post_id"].ne("") & normalized["text"].ne("")
    ].copy()
    normalized = normalized.drop_duplicates(
        subset=["provider", "post_id", "observed_at"], keep="first"
    )
    normalized = normalized.sort_values(
        ["created_at", "provider", "post_id", "observed_at"],
        na_position="last",
    ).reset_index(drop=True)
    return normalized.loc[:, CANONICAL_COLUMNS]


def load_xueqiu_export(
    path: Path | str,
    *,
    authorization_path: Path | str | None = None,
    default_observed_at: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    export_path = Path(path)
    if not export_path.exists():
        raise FileNotFoundError(export_path)
    metadata = (
        AuthorizationMetadata.from_path(authorization_path)
        if authorization_path is not None
        else None
    )
    return normalize_xueqiu_frame(
        _read_export(export_path),
        metadata=metadata,
        default_observed_at=default_observed_at,
    )


def first_observed_post_snapshot(posts: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest observable content version of each provider post."""
    if posts.empty:
        return posts.copy()
    source = posts.copy()
    for column in ("created_at", "observed_at", "content_observed_at"):
        source[column] = pd.to_datetime(source[column], errors="coerce")
    source["_effective_observed_at"] = (
        source["observed_at"]
        .fillna(source["content_observed_at"])
        .fillna(source["created_at"])
    )
    source = source.sort_values(
        [
            "provider",
            "post_id",
            "_effective_observed_at",
            "content_observed_at",
        ],
        na_position="last",
    )
    source = source.drop_duplicates(
        subset=["provider", "post_id"], keep="first"
    )
    return source.drop(columns="_effective_observed_at").reset_index(drop=True)


def append_canonical_csv(new_posts: pd.DataFrame, output_path: Path | str) -> pd.DataFrame:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [new_posts]
    if path.exists():
        existing = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        for column in (
            "created_at",
            "observed_at",
            "content_observed_at",
            "metrics_observed_at",
        ):
            if column in existing:
                existing[column] = pd.to_datetime(existing[column], errors="coerce")
        frames.insert(0, existing)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["provider", "post_id", "observed_at"], keep="last"
    )
    combined = combined.sort_values(
        ["created_at", "provider", "post_id", "observed_at"],
        na_position="last",
    ).reset_index(drop=True)
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return combined

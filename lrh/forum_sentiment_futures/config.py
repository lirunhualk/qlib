from __future__ import annotations

from dataclasses import dataclass


DEFAULT_COST_BPS_PER_SIDE = 1.0
DEFAULT_SENTIMENT_WINDOW = 60
DEFAULT_MIN_SENTIMENT_PERIODS = 20
DEFAULT_LOOKBACK_YEARS = 5
DEFAULT_OUTPUT_DIRNAME = "report"
DEFAULT_CACHE_DIRNAME = "data"


@dataclass(frozen=True)
class FutureSpec:
    fut_code: str
    spot_code: str
    display_name: str
    first_trade_date: int
    forum_board_code: str
    sentiment_symbols: tuple[str, ...]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    display_name: str
    kind: str
    panic_entry_z: float = 1.0
    excited_entry_z: float = 1.0
    low_activity_z: float = 0.25
    calm_abs_panic_z: float = 0.9
    fade_z: float = -0.8
    forum_panic_entry_z: float = 0.8
    max_forum_excitement_z: float = 0.75
    momentum_column: str = "spot_return_3d"


CSI300_SENTIMENT = (
    "SH600519",
    "SZ300750",
    "SH601318",
    "SH600036",
    "SZ000858",
    "SZ002594",
    "SH600030",
    "SH601012",
)

SSE50_SENTIMENT = (
    "SH600519",
    "SH601318",
    "SH600036",
    "SH600900",
    "SH600276",
    "SH600030",
    "SH601166",
    "SH601398",
)

CSI500_SENTIMENT = (
    "SH600584",
    "SZ002475",
    "SZ002463",
    "SH600196",
    "SZ300308",
    "SH601012",
    "SH600030",
    "SZ002594",
)

CSI1000_SENTIMENT = (
    "SZ300274",
    "SH688111",
    "SZ300124",
    "SZ300418",
    "SZ002185",
    "SH600206",
    "SZ002747",
    "SZ002463",
)


FUTURE_SPECS = {
    "IF": FutureSpec("IF", "000300.SH", "CSI 300", 20100416, "zssh000300", CSI300_SENTIMENT),
    "IC": FutureSpec("IC", "000905.SH", "CSI 500", 20150416, "zssh000905", CSI500_SENTIMENT),
    "IH": FutureSpec("IH", "000016.SH", "SSE 50", 20150416, "zssh000016", SSE50_SENTIMENT),
    "IM": FutureSpec("IM", "000852.SH", "CSI 1000", 20220722, "zssh000852", CSI1000_SENTIMENT),
}


def build_strategy_specs() -> list[StrategySpec]:
    return [
        StrategySpec(
            name="always_long_continuous",
            display_name="Always long continuous futures",
            kind="always_hold",
        ),
        StrategySpec(
            name="always_long_intraday",
            display_name="Always long intraday",
            kind="always_long",
        ),
        StrategySpec(
            name="panic_reversal_long",
            display_name="Panic reversal long",
            kind="panic_long",
        ),
        StrategySpec(
            name="excitement_short",
            display_name="Excitement short",
            kind="excitement_short",
        ),
        StrategySpec(
            name="dual_forum_sentiment",
            display_name="Dual forum sentiment",
            kind="dual",
        ),
    ]

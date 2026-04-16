from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MYSQL_DSN = "mysql+pymysql://root:pazqjj2021!@localhost:3306/aShare?charset=utf8mb4"
DEFAULT_COST_BPS_PER_SIDE = 1.0
DEFAULT_LIQUID_TOP_N = 2
DEFAULT_OUTPUT_DIRNAME = "outputs"
DEFAULT_REPORT_DIRNAME = "report"


@dataclass(frozen=True)
class FutureSpec:
    fut_code: str
    spot_code: str
    display_name: str
    first_trade_date: int


@dataclass(frozen=True)
class StrategySpec:
    name: str
    display_name: str
    family: str
    selector: str
    initial_rank: int | None = None
    roll_days_before_expiry: int | None = None
    liquid_top_n: int | None = None


FUTURE_SPECS = {
    "IF": FutureSpec("IF", "000300.SH", "沪深300", 20100416),
    "IC": FutureSpec("IC", "000905.SH", "中证500", 20150416),
    "IH": FutureSpec("IH", "000016.SH", "上证50", 20150416),
    "IM": FutureSpec("IM", "000852.SH", "中证1000", 20220722),
}


def build_strategy_specs(liquid_top_n: int = DEFAULT_LIQUID_TOP_N) -> list[StrategySpec]:
    return [
        StrategySpec("dominant_volume", "主力成交量", "dominant", "dominant_volume"),
        StrategySpec("dominant_oi", "主力持仓量", "dominant", "dominant_oi"),
        StrategySpec("max_carry_daily", "每日重选最大年化贴水", "carry", "max_carry_daily"),
        StrategySpec(
            "max_carry_daily_liquid",
            f"最大年化贴水(流动性前{liquid_top_n})",
            "carry",
            "max_carry_daily_liquid",
            liquid_top_n=liquid_top_n,
        ),
        StrategySpec("front_roll_5", "近月提前5日", "front_roll", "roll_chain", initial_rank=1, roll_days_before_expiry=5),
        StrategySpec("front_roll_3", "近月提前3日", "front_roll", "roll_chain", initial_rank=1, roll_days_before_expiry=3),
        StrategySpec("front_roll_1", "近月提前1日", "front_roll", "roll_chain", initial_rank=1, roll_days_before_expiry=1),
        StrategySpec(
            "second_roll_5",
            "次近月提前5日",
            "second_roll",
            "roll_chain",
            initial_rank=2,
            roll_days_before_expiry=5,
        ),
        StrategySpec(
            "second_roll_3",
            "次近月提前3日",
            "second_roll",
            "roll_chain",
            initial_rank=2,
            roll_days_before_expiry=3,
        ),
        StrategySpec(
            "second_roll_1",
            "次近月提前1日",
            "second_roll",
            "roll_chain",
            initial_rank=2,
            roll_days_before_expiry=1,
        ),
        StrategySpec("best_carry_hold", "最大年化贴水持有至到期", "carry_hold", "best_carry_hold"),
    ]

"""Simple offline backtester for the tutorial TOMATOES/EMERALDS data.

Loads the four CSVs under TUTORIAL_ROUND_1_DATA_IMC_4 and evaluates a
threshold-based taker strategy:
- Compute a rolling fair value from the mid prices.
- Buy when a trade prints below fair - buy_offset.
- Sell when a trade prints above fair + sell_offset.
- Enforce an absolute inventory limit (default 20).

The script sweeps a small grid of window sizes and offsets and reports the
best PnL across both training days (-2 and -1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DATA_ROOT = Path(__file__).parent.parent.parent / "TUTORIAL_ROUND_1_DATA_IMC_4"


@dataclass
class Params:
    window: int
    buy_offset: float
    sell_offset: float
    limit: int = 20


def load_prices(day: int) -> pd.DataFrame:
    prices = pd.read_csv(DATA_ROOT / f"prices_round_0_day_{day}.csv", sep=";")
    prices.columns = [c.strip().lower() for c in prices.columns]
    prices = prices.rename(columns={"product": "product", "mid_price": "mid_price"})
    prices = prices[["timestamp", "product", "mid_price"]]
    return prices


def load_trades(day: int) -> pd.DataFrame:
    trades = pd.read_csv(DATA_ROOT / f"trades_round_0_day_{day}.csv", sep=";")
    trades.columns = [c.strip().lower() for c in trades.columns]
    trades = trades[["timestamp", "symbol", "price", "quantity"]]
    trades = trades.rename(columns={"symbol": "product"})
    return trades


def calc_fair(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    prices = prices.sort_values(["product", "timestamp"]).copy()
    prices["fair"] = (
        prices.groupby("product")["mid_price"]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return prices


def simulate_day(prices: pd.DataFrame, trades: pd.DataFrame, params: Params) -> float:
    prices = calc_fair(prices, params.window)
    fair_lookup = prices.set_index(["product", "timestamp"]) ["fair"].to_dict()
    mid_lookup = prices.set_index(["product", "timestamp"]) ["mid_price"].to_dict()

    cash: dict[str, float] = {"TOMATOES": 0.0, "EMERALDS": 0.0}
    pos: dict[str, int] = {"TOMATOES": 0, "EMERALDS": 0}

    for _, row in trades.sort_values("timestamp").iterrows():
        product = row["product"]
        ts = row["timestamp"]
        price = row["price"]
        qty = int(row["quantity"])

        fair = fair_lookup.get((product, ts))
        if fair is None:
            continue

        # Simple taker rule around rolling fair.
        if price <= fair - params.buy_offset and pos[product] + qty <= params.limit:
            cash[product] -= price * qty
            pos[product] += qty
        elif price >= fair + params.sell_offset and pos[product] - qty >= -params.limit:
            cash[product] += price * qty
            pos[product] -= qty

    # Mark-to-market using last mid for each product.
    pnl = 0.0
    for product in ["TOMATOES", "EMERALDS"]:
        last_ts = prices[prices["product"] == product]["timestamp"].max()
        last_mid = mid_lookup.get((product, last_ts), 0.0)
        pnl += cash[product] + pos[product] * last_mid
    return pnl


def evaluate(params: Params, days: Iterable[int]) -> float:
    total = 0.0
    for day in days:
        prices = load_prices(day)
        trades = load_trades(day)
        total += simulate_day(prices, trades, params)
    return total


def sweep():
    days = [-2, -1]
    best: tuple[float, Params] | None = None

    for window in [10, 20, 40, 80, 120]:
        for buy_offset in [2, 3, 4, 5]:
            for sell_offset in [2, 3, 4, 5]:
                params = Params(window=window, buy_offset=buy_offset, sell_offset=sell_offset)
                pnl = evaluate(params, days)
                if best is None or pnl > best[0]:
                    best = (pnl, params)

    assert best is not None
    pnl, params = best
    print(f"Best PnL {pnl:.2f} with window={params.window}, buy_offset={params.buy_offset}, sell_offset={params.sell_offset}, limit={params.limit}")


if __name__ == "__main__":
    sweep()
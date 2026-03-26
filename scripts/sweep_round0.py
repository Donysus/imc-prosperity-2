import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
ALG_DIR = SRC_DIR / "algorithms"
for path in (BASE_DIR, SRC_DIR, ALG_DIR):
    p_str = str(path)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Make backtester datamodel available under the expected name.
import prosperity2bt.datamodel as bt_datamodel
sys.modules.setdefault("datamodel", bt_datamodel)

from prosperity2bt.file_reader import PackageResourcesReader
from prosperity2bt.runner import run_backtest

# Import the submission trader
from src.submissions.prosperity4_tutorial import Trader as SubmissionTrader


@dataclass
class TrialResult:
    config: Dict[str, Any]
    pnl: float


def sample_config(rng: random.Random) -> Dict[str, Any]:
    def uniform(a: float, b: float) -> float:
        return a + (b - a) * rng.random()

    def choice(seq):
        return seq[math.floor(rng.random() * len(seq))]

    return {
        "TOMATO": {
            "mode": choice(["momentum", "mean_revert"]),
            "trend_weight": uniform(0.4, 1.0),
            "micro_blend": uniform(0.0, 0.7),
            "micro_alpha": 0.55,
            "close_flatten_ts": choice([170000, 180000, 185000]),
            "skew_base": uniform(0.05, 0.12),
            "skew_close": uniform(0.16, 0.24),
        },
        "STARFRUIT": {
            "mode": choice(["momentum", "mean_revert"]),
            "trend_weight": uniform(0.3, 0.8),
            "micro_blend": uniform(0.0, 0.5),
            "micro_alpha": 0.55,
            "close_flatten_ts": choice([90000, 95000, 99000]),
            "skew_base": uniform(0.06, 0.14),
            "skew_close": uniform(0.14, 0.22),
        },
        "AMETHYSTS": {
            "use_popular_mid": True,
            "passive_edge": choice([1.0, 1.2, 1.5]),
        },
        "EMERALD": {
            "passive_edge": choice([1.0, 1.2, 1.5]),
        },
    }


def apply_config(trader: SubmissionTrader, cfg: Dict[str, Any]) -> None:
    # Replace only known keys to avoid surprises
    for product, params in cfg.items():
        if product not in trader.config:
            trader.config[product] = {}
        trader.config[product].update(params)


def run_trial(cfg: Dict[str, Any], reader: PackageResourcesReader) -> float:
    trader = SubmissionTrader()
    apply_config(trader, cfg)
    result = run_backtest(
        trader,
        reader,
        round_num=0,
        day_num=-2,
        print_output=False,
        disable_trades_matching=False,
        no_names=False,
        show_progress_bar=False,
    )
    # BacktestResult exposes activity logs; profit/loss per product is the last column.
    # Take the final timestamp rows (one per product) and sum their PnL.
    max_ts = max(row.timestamp for row in result.activity_logs)
    final_rows = [row for row in result.activity_logs if row.timestamp == max_ts]
    return float(sum(r.columns[-1] for r in final_rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    reader = PackageResourcesReader()

    results: List[TrialResult] = []
    best: Tuple[float, Dict[str, Any]] | None = None

    for i in range(args.trials):
        cfg = sample_config(rng)
        pnl = run_trial(cfg, reader)
        results.append(TrialResult(cfg, pnl))
        if best is None or pnl > best[0]:
            best = (pnl, cfg)
        print(f"trial {i+1}/{args.trials}: pnl={pnl:.2f}")

    results.sort(key=lambda r: r.pnl, reverse=True)

    print("\nTop 5 configs:")
    for rank, res in enumerate(results[:5], 1):
        print(f"#{rank}: pnl={res.pnl:.2f} config={res.config}")

    if best:
        print("\nBest config to apply in submission:")
        print(best[1])


if __name__ == "__main__":
    main()

"""
Dixon-Coles calibration backtest.

Fits the DC model by MLE on a training slice of a football season, then
evaluates match-outcome forecasts on a held-out test slice against
Pinnacle closing odds. Produces:

    reports/metrics.md             — markdown summary of scoring rules
    reports/reliability.png        — reliability diagram (calibration)
    reports/kelly_pnl.png          — fractional-Kelly bankroll curve
    reports/metrics.json           — machine-readable numbers

Data source: football-data.co.uk (free, redistributable CSVs). Pinnacle
closing odds columns are PSCH / PSCD / PSCA. Downloads into data_raw/.

Usage:
    python -m scripts.backtest_report --league E0 --season 2324
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running this file directly (python scripts/backtest_report.py) as
# well as via `python -m scripts.backtest_report`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import (
    devig_proportional, devig_shin,
    log_loss, brier_score, ranked_probability_score,
    expected_calibration_error, reliability_bins,
    kelly_backtest, baseline_uniform, baseline_always_home,
)
from dc_fit import fit_dixon_coles


DATA_DIR = ROOT / "data_raw"
REPORT_DIR = ROOT / "reports"
FD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def ensure_csv(league: str, season: str) -> Path:
    """Download the CSV if not already cached. Returns the local path."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{league}_{season}.csv"
    if not path.exists():
        url = FD_URL.format(league=league, season=season)
        print(f"[data] downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def load_matches(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    # Keep only rows with the fields we need
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                "PSCH", "PSCD", "PSCA"]
    df = df.dropna(subset=required)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Fit + predict
# ---------------------------------------------------------------------------

def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame, xi: float):
    fit = fit_dixon_coles(
        train["HomeTeam"].tolist(),
        train["AwayTeam"].tolist(),
        train["FTHG"].astype(int).tolist(),
        train["FTAG"].astype(int).tolist(),
        dates=train["Date"].tolist(),
        xi=xi,
    )
    probs = []
    outcomes = []
    odds = []
    skipped = 0
    for _, row in test.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        if h not in fit.atk or a not in fit.atk:
            skipped += 1
            continue
        probs.append(fit.probs(h, a))
        outcomes.append(str(row["FTR"]))
        odds.append((float(row["PSCH"]), float(row["PSCD"]), float(row["PSCA"])))
    if skipped:
        print(f"[warn] {skipped} test matches skipped (team not in train set)")
    return fit, probs, outcomes, odds


# ---------------------------------------------------------------------------
# Metrics + plots
# ---------------------------------------------------------------------------

def compute_metrics(probs, outcomes, odds):
    market = [devig_shin(*od) for od in odds]
    uniform = baseline_uniform(len(probs))
    base_rate = baseline_always_home(len(probs))

    def summary(label, p_list):
        return {
            "forecaster": label,
            "log_loss": log_loss(p_list, outcomes),
            "brier": brier_score(p_list, outcomes),
            "rps": ranked_probability_score(p_list, outcomes),
            "ece": expected_calibration_error(p_list, outcomes),
        }

    return [
        summary("Uniform (1/3, 1/3, 1/3)", uniform),
        summary("Base-rate (0.46, 0.27, 0.27)", base_rate),
        summary("Market (Pinnacle, Shin-devigged)", market),
        summary("Dixon-Coles MLE", probs),
    ]


def plot_reliability(probs, outcomes, path: Path):
    bins = reliability_bins(probs, outcomes, n_bins=10)
    xs = [b["mean_predicted"] for b in bins if b["count"] > 0]
    ys = [b["empirical_freq"] for b in bins if b["count"] > 0]
    counts = [b["count"] for b in bins if b["count"] > 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.scatter(xs, ys, s=[min(300, c) for c in counts], alpha=0.7,
               label="Dixon-Coles (size = bin count)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Reliability diagram (1X2 outcomes)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_kelly(records, starting: float, path: Path):
    bankroll = [starting] + [r.bankroll_after for r in records]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(bankroll)), bankroll, lw=2)
    ax.axhline(starting, color="k", ls=":", alpha=0.5, label=f"Starting bankroll = {starting}")
    ax.set_xlabel("Bets placed")
    ax.set_ylabel("Bankroll")
    ax.set_title("Fractional-Kelly backtest (Dixon-Coles model vs Pinnacle closing)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_markdown(metrics, kelly_stats, meta, path: Path):
    lines = [
        f"# Dixon-Coles Calibration Backtest",
        "",
        f"**League:** {meta['league']} &nbsp;&nbsp; "
        f"**Season:** {meta['season']} &nbsp;&nbsp; "
        f"**Train / Test split:** {meta['train_n']} / {meta['test_n']} matches &nbsp;&nbsp; "
        f"**Time-decay xi:** {meta['xi']}",
        "",
        "## Scoring rules on the held-out test set",
        "",
        "| Forecaster | Log loss ↓ | Brier ↓ | RPS ↓ | ECE ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['forecaster']} | {m['log_loss']:.4f} | {m['brier']:.4f} | "
            f"{m['rps']:.4f} | {m['ece']:.4f} |"
        )
    lines += [
        "",
        "Uniform = information-free baseline (ln 3 ≈ 1.0986). "
        "Market = Pinnacle closing odds after Shin de-vigging; this is the "
        "sharp-market reference. A model that beats the market out-of-sample "
        "is showing genuine edge.",
        "",
        "## Fractional-Kelly PnL (1/4 Kelly, 2% min edge)",
        "",
        f"- Starting bankroll: **{kelly_stats['starting']:.2f}**",
        f"- Final bankroll: **{kelly_stats['final']:.2f}**",
        f"- Return: **{kelly_stats['return_pct']:+.2f}%**",
        f"- Bets placed: **{kelly_stats['n_bets']}** (of {meta['test_n']} "
        f"test matches, hit rate {kelly_stats['hit_rate']:.1%})",
        "",
        "See `reliability.png` for the calibration diagram and "
        "`kelly_pnl.png` for the bankroll curve.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="E0", help="football-data.co.uk league code")
    parser.add_argument("--season", default="2324",
                        help="Season code e.g. 2324 for 2023-24")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--xi", type=float, default=0.0,
                        help="Time-decay rate per day (Dixon-Coles)")
    parser.add_argument("--kelly-frac", type=float, default=0.25)
    parser.add_argument("--min-edge", type=float, default=0.02)
    parser.add_argument("--bankroll", type=float, default=100.0)
    args = parser.parse_args()

    csv_path = ensure_csv(args.league, args.season)
    df = load_matches(csv_path)
    split = int(len(df) * args.train_frac)
    train = df.iloc[:split]
    test = df.iloc[split:]
    print(f"[fit] train={len(train)}  test={len(test)}  teams={df['HomeTeam'].nunique()}")

    fit, probs, outcomes, odds = fit_and_predict(train, test, xi=args.xi)
    print(f"[fit] converged={fit.converged}  LL={fit.log_likelihood:.2f}  "
          f"gamma={fit.home_advantage:.3f}  rho={fit.rho:.4f}")

    metrics = compute_metrics(probs, outcomes, odds)
    print("\nScoring rules:")
    for m in metrics:
        print(f"  {m['forecaster']:<40}  "
              f"LL={m['log_loss']:.4f}  Brier={m['brier']:.4f}  "
              f"RPS={m['rps']:.4f}  ECE={m['ece']:.4f}")

    records, final_bank = kelly_backtest(
        probs, outcomes, odds,
        kelly_fraction_multiplier=args.kelly_frac,
        starting_bankroll=args.bankroll,
        min_edge=args.min_edge,
    )
    n_bets = len(records)
    hit_rate = (sum(1 for r in records if r.outcome_hit) / n_bets) if n_bets else 0.0
    kelly_stats = {
        "starting": args.bankroll,
        "final": final_bank,
        "return_pct": 100.0 * (final_bank / args.bankroll - 1.0),
        "n_bets": n_bets,
        "hit_rate": hit_rate,
    }
    print(f"\n[kelly] bets={n_bets}  hit_rate={hit_rate:.1%}  "
          f"final={final_bank:.2f}  return={kelly_stats['return_pct']:+.2f}%")

    REPORT_DIR.mkdir(exist_ok=True)
    plot_reliability(probs, outcomes, REPORT_DIR / "reliability.png")
    plot_kelly(records, args.bankroll, REPORT_DIR / "kelly_pnl.png")
    meta = {
        "league": args.league, "season": args.season,
        "train_n": len(train), "test_n": len(test),
        "xi": args.xi,
    }
    write_markdown(metrics, kelly_stats, meta, REPORT_DIR / "metrics.md")
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "kelly": kelly_stats, "meta": meta}, indent=2),
        encoding="utf-8",
    )
    print(f"\nReport written to {REPORT_DIR}/")


if __name__ == "__main__":
    main()

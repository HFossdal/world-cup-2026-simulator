"""
Out-of-sample calibration of the Dixon-Coles model on real international
men's senior football matches.

Pipeline:
    1. Pull martj42/international_results (~50k matches since 1872, MIT).
    2. Slice the last `--years` of matches; chronological train/test split.
    3. Fit DC on train with per-match neutral-venue handling and DC
       exponential time decay (xi=0.0065/day).
    4. Predict (P_home, P_draw, P_away) for every test match using the
       fitted ratings — gamma applied iff the test match is non-neutral.
    5. Score against realised outcomes with log loss, Brier, RPS, ECE.
    6. Compare to two baselines:
           - Uniform (1/3, 1/3, 1/3)
           - Base rate from training outcomes
    7. Paired bootstrap (2000 resamples) on per-match log-loss differences
       to give a 95% CI on the gap (DC vs baselines).
    8. Reliability diagram + report files in reports/.

Why no market reference: international matches don't have a free
machine-readable closing-odds dataset. Validating against realised outcomes
on the same data the WC sim uses is the more honest test for this project.

Usage:
    python -m scripts.validate_international --years 4 --test-frac 0.20
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import (
    baseline_base_rate, baseline_uniform,
    brier_score, expected_calibration_error,
    log_loss, paired_bootstrap_ci, per_match_log_loss,
    ranked_probability_score, reliability_bins,
)
from dc_fit import fit_dixon_coles
from scripts.fit_international import ensure_csv, load_matches


REPORT_DIR = ROOT / "reports"


def outcome_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame, xi: float):
    fit = fit_dixon_coles(
        home=train["home_team"].tolist(),
        away=train["away_team"].tolist(),
        home_goals=train["home_score"].tolist(),
        away_goals=train["away_score"].tolist(),
        dates=train["date"].tolist(),
        xi=xi,
        neutral=train["neutral"].tolist(),
    )
    probs: list[tuple[float, float, float]] = []
    outcomes: list[str] = []
    skipped = 0
    for _, r in test.iterrows():
        h, a = r["home_team"], r["away_team"]
        if h not in fit.atk or a not in fit.atk:
            skipped += 1
            continue
        p = fit.probs(h, a, neutral=bool(r["neutral"]))
        probs.append(p)
        outcomes.append(outcome_label(int(r["home_score"]), int(r["away_score"])))
    if skipped:
        print(f"[warn] {skipped} test matches skipped (team not in train fit)")
    return fit, probs, outcomes


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
    ax.set_title("Reliability diagram (1X2 outcomes, international matches)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_markdown(meta, scores, ci, path: Path):
    lines = [
        "# Dixon-Coles out-of-sample calibration — international matches",
        "",
        f"**Data window:** {meta['window_start']} → {meta['window_end']} "
        f"({meta['n_total']} matches, {meta['neutral_share']:.0%} neutral)  ",
        f"**Train / Test split:** {meta['n_train']} / {meta['n_test']} "
        f"matches, chronological  ",
        f"**Test match teams:** {meta['n_test_teams']} "
        f"(skipped {meta['n_skipped']} matches whose teams weren't in the train fit)  ",
        f"**Fit parameters:** xi={meta['xi']}, "
        f"mu={meta['mu_dc']:.3f}, gamma={meta['gamma_dc']:.3f}, rho={meta['rho_dc']:.3f}  ",
        "",
        "## Scoring rules on the held-out test set",
        "",
        "| Forecaster | Log loss ↓ | Brier ↓ | RPS ↓ | ECE ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in scores:
        lines.append(
            f"| {s['name']} | {s['log_loss']:.4f} | {s['brier']:.4f} | "
            f"{s['rps']:.4f} | {s['ece']:.4f} |"
        )
    lines += [
        "",
        "## 95% paired-bootstrap CI on the log-loss gap (DC vs baseline)",
        "",
        "| Comparison | Δ log-loss | 95% CI |",
        "|---|---:|---|",
    ]
    for c in ci:
        sig = "**(significant)**" if (c["lo"] < 0 and c["hi"] < 0) else "(not significant)"
        lines.append(
            f"| DC − {c['baseline']} | {c['point']:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] {sig} |"
        )
    lines += [
        "",
        "Negative gap = Dixon-Coles has lower loss than the baseline "
        "(better forecast). A CI that excludes zero means the gap is "
        f"statistically distinguishable from noise at the 95% level "
        f"(n={meta['n_test']} test matches, {meta['n_resamples']} resamples).",
        "",
        "See `intl_reliability.png` for the calibration diagram.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=4.0,
                        help="Window of historical international matches to use")
    parser.add_argument("--test-frac", type=float, default=0.20,
                        help="Most recent fraction held out for testing")
    parser.add_argument("--xi", type=float, default=0.0065,
                        help="DC time-decay rate per day (0 disables)")
    parser.add_argument("--min-matches", type=int, default=5,
                        help="Drop teams with fewer than this many train matches")
    parser.add_argument("--n-resamples", type=int, default=2000)
    args = parser.parse_args()

    csv_path = ensure_csv()
    df = load_matches(csv_path, args.years)
    n_total = len(df)
    neutral_share = float(df["neutral"].mean())
    print(f"[data] {n_total} matches "
          f"({df['date'].min().date()} -> {df['date'].max().date()}), "
          f"{neutral_share:.1%} neutral")

    # Chronological split
    split = int(n_total * (1.0 - args.test_frac))
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)

    # Drop low-data teams from the train fit (their MLE est. is noisy)
    counts: dict[str, int] = {}
    for _, r in train.iterrows():
        counts[r["home_team"]] = counts.get(r["home_team"], 0) + 1
        counts[r["away_team"]] = counts.get(r["away_team"], 0) + 1
    keep = {t for t, c in counts.items() if c >= args.min_matches}
    train = train[train["home_team"].isin(keep) & train["away_team"].isin(keep)].reset_index(drop=True)
    print(f"[fit ] train={len(train)}  test={len(test)}  teams_in_fit={len(keep)}")

    fit, dc_probs, dc_outcomes = fit_and_predict(train, test, xi=args.xi)
    print(f"[fit ] converged={fit.converged}  LL={fit.log_likelihood:.1f}  "
          f"mu={fit.baseline:.3f}  gamma={fit.home_advantage:.3f}  "
          f"rho={fit.rho:.4f}")

    n_test_kept = len(dc_outcomes)
    uniform = baseline_uniform(n_test_kept)
    train_outcomes = [outcome_label(int(r["home_score"]), int(r["away_score"]))
                      for _, r in train.iterrows()]
    base = baseline_base_rate(train_outcomes, n_test_kept)

    forecasters = [
        ("Uniform (1/3, 1/3, 1/3)", uniform),
        ("Base rate (train-set frequencies)", base),
        ("Dixon-Coles MLE", dc_probs),
    ]
    scores = []
    for name, p in forecasters:
        s = {
            "name": name,
            "log_loss": log_loss(p, dc_outcomes),
            "brier": brier_score(p, dc_outcomes),
            "rps": ranked_probability_score(p, dc_outcomes),
            "ece": expected_calibration_error(p, dc_outcomes),
        }
        scores.append(s)
        print(f"  {name:<40}  LL={s['log_loss']:.4f}  Brier={s['brier']:.4f}  "
              f"RPS={s['rps']:.4f}  ECE={s['ece']:.4f}")

    # Paired bootstrap on the log-loss gap (DC vs baselines)
    dc_losses = per_match_log_loss(dc_probs, dc_outcomes)
    ci_records = []
    for name, baseline_probs in [("Uniform", uniform), ("Base rate", base)]:
        b_losses = per_match_log_loss(baseline_probs, dc_outcomes)
        point, lo, hi = paired_bootstrap_ci(dc_losses, b_losses,
                                            n_resamples=args.n_resamples)
        ci_records.append({"baseline": name, "point": point, "lo": lo, "hi": hi})
        verdict = "lower than" if (lo < 0 and hi < 0) else (
            "higher than" if (lo > 0 and hi > 0) else "indistinguishable from"
        )
        print(f"  CI(DC - {name}): {point:+.4f}  [{lo:+.4f}, {hi:+.4f}]  -> "
              f"DC log-loss is {verdict} {name}'s")

    REPORT_DIR.mkdir(exist_ok=True)
    plot_reliability(dc_probs, dc_outcomes, REPORT_DIR / "intl_reliability.png")
    meta = {
        "window_start": str(df["date"].min().date()),
        "window_end": str(df["date"].max().date()),
        "n_total": n_total,
        "neutral_share": neutral_share,
        "n_train": len(train),
        "n_test": n_test_kept,
        "n_skipped": len(test) - n_test_kept,
        "n_test_teams": len({t for t in test["home_team"]} | {t for t in test["away_team"]}),
        "xi": args.xi,
        "mu_dc": fit.baseline,
        "gamma_dc": fit.home_advantage,
        "rho_dc": fit.rho,
        "n_resamples": args.n_resamples,
    }
    write_markdown(meta, scores, ci_records, REPORT_DIR / "intl_metrics.md")
    (REPORT_DIR / "intl_metrics.json").write_text(
        json.dumps({"meta": meta, "scores": scores, "ci": ci_records}, indent=2),
        encoding="utf-8",
    )
    print(f"\nReport written to {REPORT_DIR}/intl_metrics.md")


if __name__ == "__main__":
    main()

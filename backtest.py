"""
World Cup 2026 Simulator — Backtesting & Calibration

Pure-function metrics and utilities for evaluating Dixon-Coles match
probabilities against historical outcomes and bookmaker closing odds.

Provides:
    - De-vigging (proportional, Shin) to convert market odds into fair
      implied probabilities that sum to one.
    - Multi-class log loss, Brier score, and ranked probability score
      (Epstein 1969) — the latter respects the ordinal nature of 1X2
      outcomes and is the standard metric in forecasting literature.
    - Kelly criterion bet sizing with fractional Kelly support.
    - Reliability-diagram binning for calibration plots.

Outcome convention: "H" = home win, "D" = draw, "A" = away win.
Probability convention: (p_home, p_draw, p_away), summing to 1.

References:
    Dixon & Coles (1997), "Modelling Association Football Scores..."
    Shin (1993), "Measuring the Incidence of Insider Trading..."
    Constantinou & Fenton (2012), "Solving the Problem of Inadequate
        Scoring Rules for Assessing Probabilistic Football Forecasts."
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

# Outcome labels in canonical order (ordinal: H < D < A is not meaningful,
# but the 1X2 ordering is conventional for bookmakers and RPS).
OUTCOMES: tuple[str, str, str] = ("H", "D", "A")
_OUTCOME_INDEX = {o: i for i, o in enumerate(OUTCOMES)}

# Clamp probabilities away from 0 / 1 to keep log loss finite.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# De-vigging: market odds -> fair implied probabilities
# ---------------------------------------------------------------------------

def implied_probs_raw(odds_h: float, odds_d: float, odds_a: float) -> tuple[float, float, float]:
    """Raw bookmaker-implied probabilities (with overround). Sum > 1."""
    return (1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a)


def overround(odds_h: float, odds_d: float, odds_a: float) -> float:
    """Bookmaker margin, e.g. 0.05 means a 5% overround."""
    return sum(implied_probs_raw(odds_h, odds_d, odds_a)) - 1.0


def devig_proportional(
    odds_h: float, odds_d: float, odds_a: float
) -> tuple[float, float, float]:
    """Remove overround by dividing each raw implied probability by the sum.

    Simple, fast, and the de facto retail standard. Assumes the bookmaker
    applies a uniform multiplicative margin across all outcomes — a decent
    approximation for liquid 1X2 markets.
    """
    raw = implied_probs_raw(odds_h, odds_d, odds_a)
    total = sum(raw)
    return tuple(p / total for p in raw)  # type: ignore[return-value]


def devig_shin(
    odds_h: float, odds_d: float, odds_a: float, max_iter: int = 200, tol: float = 1e-10
) -> tuple[float, float, float]:
    """Shin's method: models the overround as insider-trader protection.

    Solves for z in [0, 1) such that the fair probabilities
        p_i = (sqrt(z^2 + 4*(1-z)*pi_i^2 / S) - z) / (2*(1-z))
    sum to one, where pi_i = 1/odds_i and S = sum pi_i. Typically yields
    a slightly sharper favourite than proportional de-vigging.
    """
    raw = implied_probs_raw(odds_h, odds_d, odds_a)
    s = sum(raw)
    # Bisection on z in [0, 1)
    lo, hi = 0.0, 0.999
    for _ in range(max_iter):
        z = 0.5 * (lo + hi)
        fair = [
            (math.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / s) - z) / (2.0 * (1.0 - z))
            for pi in raw
        ]
        total = sum(fair)
        if abs(total - 1.0) < tol:
            break
        if total > 1.0:
            lo = z
        else:
            hi = z
    return tuple(fair)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------

def _clamp(p: float) -> float:
    return max(_EPS, min(1.0 - _EPS, p))


def log_loss(
    probs: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
) -> float:
    """Average negative log-likelihood of the realized outcomes under the
    forecast distribution. Lower is better. A uniform 1/3-1/3-1/3 forecast
    scores ln(3) ≈ 1.0986; a perfect forecast scores 0.
    """
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes length mismatch")
    total = 0.0
    for p, o in zip(probs, outcomes):
        idx = _OUTCOME_INDEX[o]
        total -= math.log(_clamp(p[idx]))
    return total / len(probs)


def brier_score(
    probs: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
) -> float:
    """Multi-class Brier score: mean squared error between the forecast
    probability vector and the one-hot outcome vector. Range [0, 2], lower
    is better. A uniform forecast scores 2/3.
    """
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes length mismatch")
    total = 0.0
    for p, o in zip(probs, outcomes):
        idx = _OUTCOME_INDEX[o]
        for k, pk in enumerate(p):
            y = 1.0 if k == idx else 0.0
            total += (pk - y) ** 2
    return total / len(probs)


def ranked_probability_score(
    probs: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
) -> float:
    """RPS (Epstein 1969) for ordered outcomes H < D < A. Penalizes
    probability mass that lies far from the realized outcome more than
    mass that lies nearby — the preferred metric for 1X2 forecasting
    (Constantinou & Fenton 2012). Lower is better. Range [0, 1].
    """
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes length mismatch")
    total = 0.0
    for p, o in zip(probs, outcomes):
        idx = _OUTCOME_INDEX[o]
        # Cumulative forecast and outcome distributions
        cum_f = 0.0
        cum_y = 0.0
        rps = 0.0
        for k in range(3):
            cum_f += p[k]
            cum_y += 1.0 if k == idx else 0.0
            rps += (cum_f - cum_y) ** 2
        total += rps / 2.0  # normalize by (K-1) = 2
    return total / len(probs)


# ---------------------------------------------------------------------------
# Kelly betting
# ---------------------------------------------------------------------------

def kelly_fraction(p: float, decimal_odds: float) -> float:
    """Optimal fraction of bankroll to stake for a single bet.

    f* = (p * (o - 1) - (1 - p)) / (o - 1)
       = p - (1 - p) / b,   where b = decimal_odds - 1.

    Returns 0 for non-positive-EV bets (model is not more confident than
    the market implies). Full Kelly is aggressive; multiply by 0.25-0.50
    to get fractional Kelly with lower variance.
    """
    b = decimal_odds - 1.0
    if b <= 0.0:
        return 0.0
    f = p - (1.0 - p) / b
    return max(0.0, f)


@dataclass
class BetRecord:
    stake: float
    decimal_odds: float
    outcome_hit: bool
    pnl: float
    bankroll_after: float


def kelly_backtest(
    probs: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
    odds: Sequence[tuple[float, float, float]],
    kelly_fraction_multiplier: float = 0.25,
    starting_bankroll: float = 100.0,
    min_edge: float = 0.02,
) -> tuple[list[BetRecord], float]:
    """Simulate fractional-Kelly staking on the single best-edge outcome
    per match. Returns (per-bet records, final bankroll).

    For each match we identify the outcome with the largest model-vs-market
    edge, size the bet via fractional Kelly on that outcome, and settle.
    Bets with edge < min_edge are skipped.
    """
    if not (len(probs) == len(outcomes) == len(odds)):
        raise ValueError("probs, outcomes, and odds must have equal length")
    bankroll = starting_bankroll
    records: list[BetRecord] = []
    for p, o, od in zip(probs, outcomes, odds):
        # Edge per outcome: model_prob * odds - 1
        edges = [p[k] * od[k] - 1.0 for k in range(3)]
        best = max(range(3), key=lambda k: edges[k])
        if edges[best] < min_edge:
            continue
        f_full = kelly_fraction(p[best], od[best])
        stake = bankroll * f_full * kelly_fraction_multiplier
        if stake <= 0.0:
            continue
        hit = (_OUTCOME_INDEX[o] == best)
        pnl = stake * (od[best] - 1.0) if hit else -stake
        bankroll += pnl
        records.append(BetRecord(
            stake=stake,
            decimal_odds=od[best],
            outcome_hit=hit,
            pnl=pnl,
            bankroll_after=bankroll,
        ))
    return records, bankroll


# ---------------------------------------------------------------------------
# Calibration / reliability
# ---------------------------------------------------------------------------

def reliability_bins(
    probs: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Reliability-diagram binning across all three 1X2 outcomes.

    For each bin [lo, hi), aggregates the forecasted probability mass
    assigned to a class and the empirical frequency at which that class
    was realized. Returns one dict per bin with keys:
        lo, hi, midpoint, mean_predicted, empirical_freq, count
    """
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes length mismatch")
    bins: list[dict[str, float]] = [
        {"lo": i / n_bins, "hi": (i + 1) / n_bins,
         "_pred_sum": 0.0, "_hits": 0.0, "count": 0}
        for i in range(n_bins)
    ]
    for p, o in zip(probs, outcomes):
        idx = _OUTCOME_INDEX[o]
        for k in range(3):
            pk = p[k]
            b = min(n_bins - 1, int(pk * n_bins))
            bins[b]["_pred_sum"] += pk
            bins[b]["_hits"] += 1.0 if k == idx else 0.0
            bins[b]["count"] += 1
    out: list[dict[str, float]] = []
    for b in bins:
        c = b["count"]
        if c == 0:
            mean_pred = 0.5 * (b["lo"] + b["hi"])
            emp = float("nan")
        else:
            mean_pred = b["_pred_sum"] / c
            emp = b["_hits"] / c
        out.append({
            "lo": b["lo"],
            "hi": b["hi"],
            "midpoint": 0.5 * (b["lo"] + b["hi"]),
            "mean_predicted": mean_pred,
            "empirical_freq": emp,
            "count": float(c),
        })
    return out


def expected_calibration_error(
    probs: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
    n_bins: int = 10,
) -> float:
    """Weighted mean absolute gap between forecast and realized frequency."""
    bins = reliability_bins(probs, outcomes, n_bins=n_bins)
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0
    ece = 0.0
    for b in bins:
        if b["count"] == 0:
            continue
        ece += (b["count"] / total) * abs(b["mean_predicted"] - b["empirical_freq"])
    return ece


# ---------------------------------------------------------------------------
# Baseline forecasters for sanity-check comparisons
# ---------------------------------------------------------------------------

def baseline_uniform(n: int) -> list[tuple[float, float, float]]:
    """Always predict (1/3, 1/3, 1/3). Log loss = ln 3 ≈ 1.0986."""
    return [(1 / 3, 1 / 3, 1 / 3)] * n


def baseline_always_home(n: int, p_home: float = 0.46, p_draw: float = 0.27) -> list[tuple[float, float, float]]:
    """Constant forecast at historical base rates for all matches.
    International football base rates (approx): 46% H / 27% D / 27% A."""
    p_away = 1.0 - p_home - p_draw
    return [(p_home, p_draw, p_away)] * n


def market_probs(
    odds: Iterable[tuple[float, float, float]],
    method: str = "proportional",
) -> list[tuple[float, float, float]]:
    """Convert a sequence of (odds_h, odds_d, odds_a) into fair implied probs."""
    fn = devig_shin if method == "shin" else devig_proportional
    return [fn(*od) for od in odds]

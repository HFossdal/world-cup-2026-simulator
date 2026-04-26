"""
World Cup 2026 Simulator — Forecast scoring & calibration

Pure-function metrics for evaluating Dixon-Coles match probabilities against
realised match outcomes. Used by `scripts/validate_international.py` to
assess out-of-sample calibration of the model on real international data.

Provides:
    - Multi-class log loss, Brier score, ranked probability score
      (Epstein 1969) — the latter respects the ordinal nature of 1X2
      outcomes and is the standard metric in football-forecasting literature.
    - Reliability-diagram binning + expected calibration error (ECE).
    - Paired bootstrap on per-match log-loss differences for confidence
      intervals on the gap between two forecasters.
    - Two reference baselines (uniform, base-rate) for sanity checks.

Outcome convention: "H" = home win, "D" = draw, "A" = away win.
Probability convention: (p_home, p_draw, p_away), summing to 1. For
neutral-venue international matches the "home"/"away" labels are
positional only — the DC fit is asked to predict with gamma=0.

References:
    Dixon & Coles (1997), "Modelling Association Football Scores..."
    Epstein (1969), "A Scoring System for Probability Forecasts..."
    Constantinou & Fenton (2012), "Solving the Problem of Inadequate
        Scoring Rules for Assessing Probabilistic Football Forecasts."
"""

from __future__ import annotations

import math
import random
from typing import Sequence

OUTCOMES: tuple[str, str, str] = ("H", "D", "A")
_OUTCOME_INDEX = {o: i for i, o in enumerate(OUTCOMES)}

# Clamp probabilities away from 0 / 1 to keep log loss finite.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Scoring rules (per-match contributions exposed for bootstrapping)
# ---------------------------------------------------------------------------

def _clamp(p: float) -> float:
    return max(_EPS, min(1.0 - _EPS, p))


def per_match_log_loss(
    probs: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
) -> list[float]:
    """Per-match -log P(realised). Mean of this is `log_loss`."""
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes length mismatch")
    return [-math.log(_clamp(p[_OUTCOME_INDEX[o]])) for p, o in zip(probs, outcomes)]


def log_loss(
    probs: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
) -> float:
    """Average negative log-likelihood of realised outcomes under the
    forecast distribution. Lower is better. A uniform 1/3-1/3-1/3 forecast
    scores ln(3) ≈ 1.0986; a perfect forecast scores 0.
    """
    losses = per_match_log_loss(probs, outcomes)
    return sum(losses) / len(losses)


def per_match_brier(
    probs: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
) -> list[float]:
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes length mismatch")
    out = []
    for p, o in zip(probs, outcomes):
        idx = _OUTCOME_INDEX[o]
        s = sum((p[k] - (1.0 if k == idx else 0.0)) ** 2 for k in range(3))
        out.append(s)
    return out


def brier_score(
    probs: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
) -> float:
    """Multi-class Brier: mean squared error between forecast vector and
    one-hot outcome. Range [0, 2], lower is better. Uniform scores 2/3.
    """
    vals = per_match_brier(probs, outcomes)
    return sum(vals) / len(vals)


def per_match_rps(
    probs: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
) -> list[float]:
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes length mismatch")
    out = []
    for p, o in zip(probs, outcomes):
        idx = _OUTCOME_INDEX[o]
        cum_f = 0.0
        cum_y = 0.0
        rps = 0.0
        for k in range(3):
            cum_f += p[k]
            cum_y += 1.0 if k == idx else 0.0
            rps += (cum_f - cum_y) ** 2
        out.append(rps / 2.0)  # normalize by (K - 1)
    return out


def ranked_probability_score(
    probs: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
) -> float:
    """RPS (Epstein 1969) for ordered outcomes H < D < A. Penalises
    probability mass that lies far from the realised outcome more than
    mass that lies nearby — preferred metric for 1X2 forecasting
    (Constantinou & Fenton 2012). Lower is better, range [0, 1].
    """
    vals = per_match_rps(probs, outcomes)
    return sum(vals) / len(vals)


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
    was realised. Returns one dict per bin with keys:
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
    """Weighted mean absolute gap between forecast and realised frequency."""
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
# Paired bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    losses_a: Sequence[float],
    losses_b: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap CI on (mean(losses_a) - mean(losses_b)).

    Returns (point_estimate, ci_lo, ci_hi). Negative = forecaster A has
    lower loss than B, on average. If the CI excludes zero, the gap is
    statistically meaningful at the chosen alpha.
    """
    if len(losses_a) != len(losses_b):
        raise ValueError("paired arrays must be the same length")
    n = len(losses_a)
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = [a - b for a, b in zip(losses_a, losses_b)]
    point = sum(diffs) / n
    rng = random.Random(seed)
    boot = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        boot.append(sum(diffs[i] for i in idx) / n)
    boot.sort()
    lo = boot[int(alpha / 2 * n_resamples)]
    hi = boot[int((1 - alpha / 2) * n_resamples) - 1]
    return point, lo, hi


# ---------------------------------------------------------------------------
# Reference baselines for sanity-check comparisons
# ---------------------------------------------------------------------------

def baseline_uniform(n: int) -> list[tuple[float, float, float]]:
    """Always predict (1/3, 1/3, 1/3). Log loss = ln 3 ≈ 1.0986."""
    return [(1 / 3, 1 / 3, 1 / 3)] * n


def baseline_base_rate(
    outcomes_train: Sequence[str],
    n_test: int,
) -> list[tuple[float, float, float]]:
    """Constant forecast at the empirical (H, D, A) frequencies of the
    training outcomes. A non-trivial baseline a model must beat to claim
    any predictive content beyond knowing the class priors.
    """
    counts = [0, 0, 0]
    for o in outcomes_train:
        counts[_OUTCOME_INDEX[o]] += 1
    total = sum(counts) or 1
    p = (counts[0] / total, counts[1] / total, counts[2] / total)
    return [p] * n_test

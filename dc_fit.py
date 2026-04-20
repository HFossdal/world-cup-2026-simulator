"""
Dixon-Coles (1997) maximum-likelihood fitter.

Given a training set of match rows (home, away, home_goals, away_goals, date),
fits per-team attack/defense strengths, home advantage, baseline goal rate,
and the low-score correlation parameter rho by L-BFGS-B on the joint
log-likelihood of the DC bivariate distribution.

Goal-rate parameterization (conventional DC):
    lambda_home = exp(mu + atk_home - def_away + gamma)
    lambda_away = exp(mu + atk_away - def_home)

Identifiability is fixed by centering (atk) and (def) to zero mean after
the optimizer returns.

Optional time-weighting follows Dixon-Coles:
    w(t) = exp(-xi * (t_ref - t)) in days,
with t_ref = max date in the training set. Setting xi = 0 disables weighting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


# Rho is kept inside this range during optimization — values outside can
# make the DC tau factor non-positive on the low-score cells.
_RHO_BOUNDS = (-0.30, 0.30)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DixonColesFit:
    teams: list[str]
    atk: dict[str, float]
    defn: dict[str, float]
    home_advantage: float
    baseline: float
    rho: float
    log_likelihood: float
    n_matches: int
    converged: bool = True
    message: str = ""

    def lambda_pair(self, home: str, away: str) -> tuple[float, float]:
        """Return (lambda_home, lambda_away) for a match between the given teams."""
        lam_h = math.exp(
            self.baseline + self.atk[home] - self.defn[away] + self.home_advantage
        )
        lam_a = math.exp(self.baseline + self.atk[away] - self.defn[home])
        return lam_h, lam_a

    def probs(self, home: str, away: str, max_goals: int = 10) -> tuple[float, float, float]:
        """(P_home, P_draw, P_away) from the fitted DC joint PMF."""
        lam_h, lam_a = self.lambda_pair(home, away)
        return _dc_probs(lam_h, lam_a, self.rho, max_goals=max_goals)


# ---------------------------------------------------------------------------
# Joint-PMF helpers (vectorized for speed inside the likelihood)
# ---------------------------------------------------------------------------

def _poisson_logpmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return k * np.log(lam) - lam - _logfact(k)


def _logfact(k: np.ndarray) -> np.ndarray:
    # Stable for k in {0,...,20}; our data is capped well below.
    out = np.zeros_like(k, dtype=float)
    for i in range(1, int(k.max()) + 1):
        out += (k >= i) * math.log(i)
    return out


def _dc_log_tau(x: np.ndarray, y: np.ndarray, lam_x: np.ndarray, lam_y: np.ndarray,
                rho: float) -> np.ndarray:
    """Per-match log(tau). Returns 0 on cells where (x,y) is outside {0,1}^2."""
    tau = np.ones_like(lam_x, dtype=float)
    m00 = (x == 0) & (y == 0); tau = np.where(m00, 1.0 - lam_x * lam_y * rho, tau)
    m01 = (x == 0) & (y == 1); tau = np.where(m01, 1.0 + lam_x * rho, tau)
    m10 = (x == 1) & (y == 0); tau = np.where(m10, 1.0 + lam_y * rho, tau)
    m11 = (x == 1) & (y == 1); tau = np.where(m11, 1.0 - rho, tau)
    # Guard rail: log of a non-positive number would poison the fit.
    tau = np.clip(tau, 1e-12, None)
    return np.log(tau)


def _dc_probs(lam_h: float, lam_a: float, rho: float, max_goals: int = 10
              ) -> tuple[float, float, float]:
    """Analytic (P_home, P_draw, P_away) under a fitted DC model."""
    grid_home = np.arange(max_goals + 1)
    pmf_h = np.exp(
        grid_home * math.log(lam_h) - lam_h - np.array([math.lgamma(k + 1) for k in grid_home])
    )
    pmf_a = np.exp(
        grid_home * math.log(lam_a) - lam_a - np.array([math.lgamma(k + 1) for k in grid_home])
    )
    i = grid_home[:, None]
    j = grid_home[None, :]
    lam_h_arr = np.full_like(i, lam_h, dtype=float)
    lam_a_arr = np.full_like(j, lam_a, dtype=float)
    log_tau = _dc_log_tau(i, j, lam_h_arr, lam_a_arr, rho)
    joint = np.exp(log_tau) * pmf_h[:, None] * pmf_a[None, :]
    joint /= joint.sum()
    home = float(joint[i > j].sum())
    draw = float(joint[i == j].sum())
    away = float(joint[i < j].sum())
    return home, draw, away


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_dixon_coles(
    home: Sequence[str],
    away: Sequence[str],
    home_goals: Sequence[int],
    away_goals: Sequence[int],
    dates: Sequence[datetime] | None = None,
    xi: float = 0.0,
    verbose: bool = False,
) -> DixonColesFit:
    """Fit DC parameters by MLE. Returns a DixonColesFit.

    xi = time-decay rate per day (Dixon & Coles used ~0.0065). 0 disables.
    """
    home = list(home); away = list(away)
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)
    n = len(home)
    if not (len(away) == n and len(hg) == n and len(ag) == n):
        raise ValueError("home, away, home_goals, away_goals must have equal length")

    teams = sorted(set(home) | set(away))
    team_idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    h_i = np.array([team_idx[t] for t in home])
    a_i = np.array([team_idx[t] for t in away])

    # Time weights
    if xi > 0.0 and dates is not None:
        t_ref = max(dates)
        w = np.array([math.exp(-xi * (t_ref - d).days) for d in dates])
    else:
        w = np.ones(n)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        atk = params[:T]
        defn = params[T:2 * T]
        mu = params[2 * T]
        gamma = params[2 * T + 1]
        rho = params[2 * T + 2]
        return atk, defn, mu, gamma, rho

    def neg_log_lik(params: np.ndarray) -> float:
        atk, defn, mu, gamma, rho = unpack(params)
        lam_h = np.exp(mu + atk[h_i] - defn[a_i] + gamma)
        lam_a = np.exp(mu + atk[a_i] - defn[h_i])
        # Poisson log-likelihood per match
        ll_poisson = (hg * np.log(lam_h) - lam_h
                      + ag * np.log(lam_a) - lam_a)
        # Tau correction for (0-0, 0-1, 1-0, 1-1) cells
        log_tau = _dc_log_tau(hg, ag, lam_h, lam_a, rho)
        ll = w * (ll_poisson + log_tau)
        return -float(ll.sum())

    # Initial guess: atk = def = 0, mu = log(avg goals), gamma = 0.3, rho = -0.10
    avg = float(np.mean(np.concatenate([hg, ag])))
    x0 = np.concatenate([
        np.zeros(T),                      # atk
        np.zeros(T),                      # defn
        [math.log(max(0.1, avg))],       # mu
        [0.3],                            # gamma (home advantage, ~1.35x)
        [-0.10],                          # rho
    ])

    # Bounds: atk/defn loose (-3, 3); mu loose; gamma (-0.5, 1); rho bounded
    bounds = (
        [(-3.0, 3.0)] * T
        + [(-3.0, 3.0)] * T
        + [(-5.0, 5.0)]
        + [(-0.5, 1.0)]
        + [_RHO_BOUNDS]
    )

    result = minimize(
        neg_log_lik, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "disp": verbose},
    )
    atk, defn, mu, gamma, rho = unpack(result.x)

    # Center atk and defn around zero for identifiability; push the shift
    # into the baseline mu so the fitted lambdas are unchanged.
    atk_mean = float(atk.mean())
    defn_mean = float(defn.mean())
    atk = atk - atk_mean
    defn = defn - defn_mean
    mu = float(mu) + atk_mean - defn_mean

    atk_map = {t: float(atk[i]) for t, i in team_idx.items()}
    defn_map = {t: float(defn[i]) for t, i in team_idx.items()}

    return DixonColesFit(
        teams=teams,
        atk=atk_map,
        defn=defn_map,
        home_advantage=float(gamma),
        baseline=mu,
        rho=float(rho),
        log_likelihood=-float(result.fun),
        n_matches=n,
        converged=bool(result.success),
        message=str(result.message),
    )

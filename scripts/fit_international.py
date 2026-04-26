"""
Fit Dixon-Coles attack/defense ratings on real international match data.

Pulls martj42/international_results (every men's senior international match
since 1872, MIT-licensed, redistributable), filters to the most recent N
years, fits the DC model with per-match neutral-venue handling and Dixon-
Coles exponential time decay, and writes a JSON file of fitted ratings for
all 48 World Cup 2026 finalists.

Output: data_raw/intl_results.csv (cached)
        team_ratings.json         (consumed by data.py at import time)

Usage:
    python -m scripts.fit_international --years 4 --xi 0.0065
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dc_fit import fit_dixon_coles


DATA_DIR = ROOT / "data_raw"
INTL_CSV = DATA_DIR / "intl_results.csv"
INTL_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
RATINGS_OUT = ROOT / "team_ratings.json"


# ---------------------------------------------------------------------------
# Map martj42 country names -> our 48-team FIFA-style 3-letter codes.
# Every code in data.TEAMS that doesn't equal a default English name needs
# an explicit entry here. Missing names just won't be lifted into the JSON.
# ---------------------------------------------------------------------------
NAME_TO_CODE: dict[str, str] = {
    "Mexico": "MEX",
    "South Korea": "KOR",
    "Korea Republic": "KOR",
    "South Africa": "RSA",
    "Denmark": "DEN",
    "Canada": "CAN",
    "Switzerland": "SUI",
    "Qatar": "QAT",
    "Italy": "ITA",
    "Brazil": "BRA",
    "Morocco": "MAR",
    "Haiti": "HAI",
    "Scotland": "SCO",
    "United States": "USA",
    "Paraguay": "PAR",
    "Australia": "AUS",
    "Turkey": "TUR",
    "Türkiye": "TUR",
    "Germany": "GER",
    "Curaçao": "CUR",
    "Curacao": "CUR",
    "Ivory Coast": "CIV",
    "Côte d'Ivoire": "CIV",
    "Ecuador": "ECU",
    "Netherlands": "NED",
    "Japan": "JPN",
    "Poland": "POL",
    "Tunisia": "TUN",
    "Belgium": "BEL",
    "Egypt": "EGY",
    "Iran": "IRN",
    "New Zealand": "NZL",
    "Spain": "ESP",
    "Uruguay": "URU",
    "Saudi Arabia": "KSA",
    "Cape Verde": "CPV",
    "Cabo Verde": "CPV",
    "France": "FRA",
    "Senegal": "SEN",
    "Norway": "NOR",
    "Iraq": "IRQ",
    "Argentina": "ARG",
    "Algeria": "ALG",
    "Austria": "AUT",
    "Jordan": "JOR",
    "Portugal": "POR",
    "Colombia": "COL",
    "Uzbekistan": "UZB",
    "DR Congo": "COD",
    "Democratic Republic of the Congo": "COD",
    "England": "ENG",
    "Croatia": "CRO",
    "Ghana": "GHA",
    "Panama": "PAN",
    "Northern Ireland": "NIR",
    "Wales": "WAL",
    "Bosnia and Herzegovina": "BIH",
    "Ukraine": "UKR",
    "Sweden": "SWE",
    "Albania": "ALB",
    "Romania": "ROU",
    "Slovakia": "SVK",
    "Kosovo": "XKX",
    "North Macedonia": "MKD",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Republic of Ireland": "IRL",
    "Ireland": "IRL",
    "New Caledonia": "NCL",
    "Jamaica": "JAM",
    "Bolivia": "BOL",
    "Suriname": "SUR",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def ensure_csv() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    if not INTL_CSV.exists():
        print(f"[data] downloading {INTL_URL}")
        urllib.request.urlretrieve(INTL_URL, INTL_CSV)
    return INTL_CSV


def load_matches(path: Path, years: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team",
                           "home_score", "away_score"])
    cutoff = df["date"].max() - timedelta(days=int(years * 365.25))
    df = df[df["date"] >= cutoff].reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool) if "neutral" in df.columns else False
    return df


# ---------------------------------------------------------------------------
# Form: rolling last-N points-per-match (W=1, D=0.5, L=0)
# ---------------------------------------------------------------------------

def compute_form(df: pd.DataFrame, n_window: int = 10) -> dict[str, float]:
    """For each team in the dataset, return its average points-per-match
    over its most recent `n_window` matches in the dataset."""
    rows = []
    for _, r in df.iterrows():
        h, a, hg, ag, d = r["home_team"], r["away_team"], r["home_score"], r["away_score"], r["date"]
        if hg > ag:
            ph, pa = 1.0, 0.0
        elif hg < ag:
            ph, pa = 0.0, 1.0
        else:
            ph = pa = 0.5
        rows.append((d, h, ph))
        rows.append((d, a, pa))
    rows.sort(key=lambda x: x[0])
    by_team: dict[str, list[float]] = defaultdict(list)
    for _, t, p in rows:
        by_team[t].append(p)
    return {t: float(np.mean(pts[-n_window:])) for t, pts in by_team.items()}


# ---------------------------------------------------------------------------
# Mapping fit -> simulator multiplicative scale
# ---------------------------------------------------------------------------
# Simulator formula in simulation.calculate_expected_goals:
#     lambda = AVG_GOALS_PER_TEAM * (att/1.40) * (1.40/def_opp) * form_factor
# DC neutral lambda (gamma=0 in WC sim) is:
#     lambda = exp(mu) * exp(atk - defn_opp)
# Setting att_sim = 1.40 * exp(atk_dc) and def_sim = 1.40 * exp(defn_dc)
# makes the ratios identical; the absolute scale is set by AVG_GOALS_PER_TEAM
# (1.35) vs exp(mu_dc). For international football mu_dc is empirically near
# log(1.35) ≈ 0.30, so the discrepancy is small (<5% on average lambdas).
# The exact mu_dc is also written to the JSON in case the simulator wants
# to override AVG_GOALS_PER_TEAM precisely.
SIM_BASELINE = 1.40


def to_sim_scale(atk_dc: float) -> float:
    return SIM_BASELINE * math.exp(atk_dc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=4.0,
                        help="Window of historical matches to fit on")
    parser.add_argument("--xi", type=float, default=0.0065,
                        help="DC time-decay rate per day (0 disables)")
    parser.add_argument("--form-window", type=int, default=10,
                        help="Rolling window for form computation")
    parser.add_argument("--min-matches", type=int, default=5,
                        help="Drop teams with fewer than this many matches")
    args = parser.parse_args()

    csv_path = ensure_csv()
    df = load_matches(csv_path, args.years)
    print(f"[fit] {len(df)} matches across {df['home_team'].nunique() + df['away_team'].nunique()} team-slots, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"[fit] neutral-venue share: {df['neutral'].mean():.1%}")

    # Drop teams with very few matches (their MLE estimates will be noisy)
    counts: dict[str, int] = defaultdict(int)
    for _, r in df.iterrows():
        counts[r["home_team"]] += 1
        counts[r["away_team"]] += 1
    keep_teams = {t for t, c in counts.items() if c >= args.min_matches}
    df = df[df["home_team"].isin(keep_teams) & df["away_team"].isin(keep_teams)].reset_index(drop=True)
    print(f"[fit] after min-matches={args.min_matches}: {len(df)} matches, "
          f"{len(keep_teams)} teams")

    fit = fit_dixon_coles(
        home=df["home_team"].tolist(),
        away=df["away_team"].tolist(),
        home_goals=df["home_score"].tolist(),
        away_goals=df["away_score"].tolist(),
        dates=df["date"].tolist(),
        xi=args.xi,
        neutral=df["neutral"].tolist(),
    )
    print(f"[fit] converged={fit.converged}  LL={fit.log_likelihood:.1f}  "
          f"mu={fit.baseline:.3f} (exp={math.exp(fit.baseline):.3f})  "
          f"gamma={fit.home_advantage:.3f}  rho={fit.rho:.4f}")

    form = compute_form(df, n_window=args.form_window)

    # Lift fitted values into our 48 codes
    from data import TEAMS
    ratings: dict[str, dict] = {}
    missing: list[str] = []
    for code in TEAMS:
        # Find any martj42 name that maps to this code and is in the fit
        names = [n for n, c in NAME_TO_CODE.items() if c == code]
        hit = next((n for n in names if n in fit.atk), None)
        if hit is None:
            missing.append(code)
            continue
        atk_dc = fit.atk[hit]
        defn_dc = fit.defn[hit]
        ratings[code] = {
            "attack": round(to_sim_scale(atk_dc), 4),
            "defense": round(to_sim_scale(defn_dc), 4),
            "midfield": round(0.5 * (to_sim_scale(atk_dc) + to_sim_scale(defn_dc)), 4),
            "form": round(form.get(hit, 0.5), 4),
            "atk_dc": round(atk_dc, 4),
            "defn_dc": round(defn_dc, 4),
            "n_matches_in_fit": int(counts[hit]),
            "source_name": hit,
        }
    print(f"[fit] mapped {len(ratings)}/{len(TEAMS)} WC teams; missing: {missing}")

    out = {
        "meta": {
            "fitted_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "data_window_years": args.years,
            "xi": args.xi,
            "form_window": args.form_window,
            "n_matches": len(df),
            "n_teams_in_fit": len(fit.teams),
            "log_likelihood": fit.log_likelihood,
            "mu_dc": fit.baseline,
            "gamma_dc": fit.home_advantage,
            "rho_dc": fit.rho,
            "implied_baseline_goals": math.exp(fit.baseline),
            "sim_baseline_constant": SIM_BASELINE,
            "note": "attack/defense are on the simulator's multiplicative "
                    "scale (1.40 = average). atk_dc/defn_dc are the raw "
                    "log-scale fitted values for diagnostics.",
        },
        "ratings": ratings,
    }
    RATINGS_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {RATINGS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

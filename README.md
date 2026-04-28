# World Cup 2026 Simulator ⚽️
**Probabilistic World Cup forecasting with a Dixon-Coles match engine fit on real international match data, an out-of-sample calibration validation, and an AI scenario chat.**

A Streamlit app + reproducible quant pipeline that:
- simulates the full 48-team tournament (single run + bracket, or Monte Carlo),
- models scorelines with the **Dixon-Coles bivariate distribution** (not independent Poisson),
- fits per-team attack/defense strengths by **MLE on ~4 000 real international matches** since the 2022 World Cup (`scripts/fit_international.py`, data from martj42/international_results, MIT-licensed),
- validates the model **out-of-sample** on held-out international matches with log loss, Brier, RPS, ECE, and **paired-bootstrap CIs** vs uniform / base-rate baselines (`scripts/validate_international.py`),
- exposes an AI scenario chat for "what-if" questions,
- supports optional voice narration via ElevenLabs.

> No market/odds data is used. International tournament closing odds are not freely redistributed in machine-readable form, so the project validates the model against **realised match outcomes** on the same data domain it predicts on, rather than against a market reference borrowed from an unrelated competition.

---

## Table of Contents
- [Features](#features)
- [Quickstart](#quickstart)
- [API Keys (Optional)](#api-keys-optional)
- [How to Use](#how-to-use)
- [Scenario Examples](#scenario-examples)
- [How Scenarios Work (Under the Hood)](#how-scenarios-work-under-the-hood)
- [How the Simulation Works](#how-the-simulation-works)
- [Calibration Validation](#calibration-validation)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Roadmap Ideas](#roadmap-ideas)
- [Author](#author)

---

## Features

### 🧩 Interactive Setup
- All 48 qualifiers are final as of April 2026. Defaults reproduce the actual field (including Bosnia's upset of Italy, Sweden over Poland, Czech Republic over Denmark in the UEFA playoffs).
- The 6 playoff slots remain editable so you can run **alternate-reality what-ifs** — e.g. "what if Italy had qualified instead of Bosnia?"

### 🤖 Scenario Chat (AI + fallback)
- If you set `MISTRAL_API_KEY`, Mistral interprets your prompt and applies structured modifications automatically.
- If you **don’t** set the key, the app still runs with a **basic keyword fallback parser** (useful for simple injuries/boosts/sim requests).

### 🏆 Single Tournament Run + Bracket
- Simulate the full 48-team tournament (groups → Round of 32 → final).
- View a clean SVG bracket and group tables.

### 📊 Monte Carlo Mode
- Run **100 / 500 / 1000 / 5000** tournament simulations.
- See:
  - win % (top chart),
  - stage probabilities (Group Exit → Winner),
  - “Most likely final” matchup.

### 🔊 Optional Voice Narration (ElevenLabs)
- If `ELEVENLABS_API_KEY` is set, results can be narrated via TTS.
- Includes a mute toggle in the UI.

---

## Quickstart

### 1) Install
```bash
git clone https://github.com/hfossdal/world-cup-2026-simulator
cd world-cup-2026-simulator

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2) Fit team ratings (one-time, ~10s)
```bash
python -m scripts.fit_international --years 4 --xi 0.0065
# downloads ~4 000 international matches, fits Dixon-Coles by MLE,
# writes team_ratings.json (gitignored). data.py loads it at import time.
```
If you skip this step the app still runs, but the simulator falls back to a small set of hand-typed strength multipliers in `data.py` instead of the MLE-fitted ratings.

### 3) (Optional) Configure environment
```bash
cp .env.example .env
# then edit .env to add MISTRAL_API_KEY / ELEVENLABS_API_KEY if you want them
```

### 4) Run
```bash
streamlit run app.py
```

### 5) (Optional) Re-validate the model
```bash
python -m scripts.validate_international --years 4 --test-frac 0.20
# writes reports/intl_metrics.md + reports/intl_reliability.png
```

---

## API Keys (Optional)
You can run the app **without** any API keys.

| Variable | Required? | What it enables |
|---|---:|---|
| `MISTRAL_API_KEY` | No | AI scenario understanding + structured scenario actions |
| `ELEVENLABS_API_KEY` | No | Voice narration (text-to-speech) |

Key sources (official dashboards):
- Mistral Console: https://console.mistral.ai/
- ElevenLabs: https://elevenlabs.io/

`.env.example` looks like:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

---

## How to Use

### 1) Setup screen (first time)
- Fill the **6 undecided slots** via dropdowns.
- Click **Start Simulating**.

### 2) Simulate from the sidebar
- **🏆 Simulate Tournament** → one full tournament + bracket
- **📊 Run N Simulations** → Monte Carlo probabilities

### 3) Use Scenario Chat
Type what-if prompts in the chat (or click an example chip).  
Your scenario can:
- adjust team strength,
- lock match results,
- force winners / force eliminations in specific knockout rounds,
- run a simulation immediately,
- reset back to baseline.

### 4) Reset / Change lineup
- **🔄 Reset All Modifications** clears scenario changes, locked results, and constraints.
- **Change Lineup** returns to the setup screen.

---

## Scenario Examples
Try these:
- “What if Norway wins all their group games?”
- “Simulate 1000 runs and show the top 10 winners.”
- “France loses Mbappé to injury.”
- “Brazil gets weakened by 20%.”
- “All favorites lose in the Round of 16.”
- “Lock Norway 2–1 France in the group stage, then simulate.”

Tip: If you mention a country explicitly, the UI will try to **highlight that team** in the bracket after a single-run simulation.

---

## How Scenarios Work (Under the Hood)

### With Mistral enabled
The agent instructs Mistral to return JSON actions in a fenced block, e.g.:
```json
[
  {"action":"nerf_team","team":"BRA","pct":20},
  {"action":"simulate","mode":"monte_carlo","n":1000}
]
```

Supported action types include:
- `adjust_team_rating` (attack/defense/midfield delta)
- `boost_team` / `nerf_team` (percentage applied to all ratings)
- `lock_result` (force a specific scoreline for a matchup)
- `force_group_winner` (force a team to finish 1st in their group)
- `force_round_exit` (force a team to lose in R32/R16/QF/SF/Final)
- `simulate` (`once` or `monte_carlo`)
- `reset`

### Without Mistral (fallback mode)
A simpler keyword parser kicks in. It won’t understand everything, but it can still:
- interpret basic “injury” prompts (reduces attack/defense),
- handle simple boost/nerf phrasing,
- trigger simulation when asked.

---

## How the Simulation Works
- **Match engine:** Dixon-Coles (1997) bivariate distribution — a Poisson model with a low-score correlation correction (parameter `rho`) that fixes the well-known under-prediction of 0-0, 1-1, 1-0 and 0-1 scorelines in football. The joint PMF is built over `{0..10}^2` and sampled by inverse CDF.
- **Neutral venue, always.** Every World Cup match is played on neutral ground, so the simulator applies **no home-advantage term** — no `gamma`, no home/away asymmetry.
- **Team strength is fitted, not eyeballed.** `atk_dc` / `defn_dc` come from a Dixon-Coles MLE fit on **~4 000 real international matches** since the 2022 World Cup (martj42/international_results, MIT-licensed). The fitter uses Dixon-Coles exponential time-decay (`xi=0.0065`/day, half-life ~107 days) so recent form weighs more, and a per-match **neutral-venue flag** so the home-advantage `gamma` is estimated from qualifiers only and is then **discarded** at simulator time.
- **Expected goals = the DC formula, directly.** The simulator computes `lambda_a = exp(mu + atk_a - defn_b)` and symmetrically for `lambda_b`, using the same fitted parameters that `validate_international.py` evaluates. No `AVG_GOALS_PER_TEAM` constant, no `1.40` normalisation, no separate form multiplier — all of which would double-count recency that is already in the time-decayed fit. The published calibration result therefore describes the actual simulator, not a drifted approximation.
- **`form` field is computed for narrative use** (each team's average points-per-match over its last 10 international results, W=1/D=0.5/L=0), so AI commentary and UI labels can talk about "in form" / "out of form" — but it is **not** used in the lambda calculation when fitted ratings are loaded.
- **Group standings:** points → goal difference → goals for → FIFA ranking (final tie-breaker).
- **48-team format:** 12 groups, top 2 qualify + **8 best third-place** teams.
- **Knockouts:** no draws — extra time (Dixon-Coles-sampled) + penalties if still level.
- **Head-to-head nudges** are layered on for a handful of classic matchups (see `data.HEAD_TO_HEAD`).

Re-fitting the ratings (run after pulling new match data):
```bash
python -m scripts.fit_international --years 4 --xi 0.0065
# writes team_ratings.json, which data.py loads at import time
```

All logic lives in `simulation.py`; format/bracket slots and ratings in `data.py`.

## Calibration Validation

The file `scripts/validate_international.py` is the quant-evaluation side of the project. It answers the only question that matters for a probabilistic forecaster: *is this model's distribution over outcomes actually calibrated when judged on real, held-out international matches?*

What it does:
1. Pulls `martj42/international_results` (every men's senior international since 1872, MIT-licensed, redistributable).
2. Slices the last `--years` of matches; chronological **train / test** split (no lookahead).
3. Fits the full Dixon-Coles model on the train split by **MLE** (L-BFGS-B) — per-team attack / defense, home-advantage `gamma`, baseline rate `mu`, low-score correction `rho` — with **Dixon-Coles exponential time decay** (`xi=0.0065`/day) and a per-match **neutral-venue flag** so `gamma` is estimated from qualifiers and not contaminated by tournament fixtures.
4. Predicts (P_home, P_draw, P_away) for every test match. Neutral test matches predict with `gamma=0`; non-neutral test matches predict with the fitted `gamma`.
5. Scores three forecasters with **log loss, Brier, RPS, ECE**:
   - Uniform (1/3, 1/3, 1/3) — information-free baseline
   - Base rate from training-set frequencies — beating this requires actual predictive content
   - **Dixon-Coles MLE**
6. Runs a **2 000-resample paired bootstrap** on per-match log-loss differences (DC vs each baseline) and reports a 95% CI on the gap. A CI that excludes zero means the gap is statistically meaningful, not noise.
7. Writes:
   - `reports/intl_metrics.md` — table + commentary
   - `reports/intl_reliability.png` — calibration diagram
   - `reports/intl_metrics.json` — machine-readable

Run it:
```bash
python -m scripts.validate_international --years 4 --test-frac 0.20 --xi 0.0065
```

Sample output on the last 4 years of international matches (3 234 train / 826 test, ~34% neutral venue):

| Forecaster | Log loss ↓ | Brier ↓ | RPS ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| Uniform (1/3, 1/3, 1/3) | 1.099 | 0.667 | 0.239 | 0.000 |
| Base rate (train frequencies) | 1.057 | 0.638 | 0.229 | 0.003 |
| **Dixon-Coles MLE** | **0.868** | **0.516** | **0.170** | 0.018 |

| Comparison | Δ log-loss | 95% CI (paired bootstrap, 2 000 resamples) |
|---|---:|---|
| DC − Uniform | −0.230 | [−0.272, −0.188] **(significant)** |
| DC − Base rate | −0.188 | [−0.227, −0.148] **(significant)** |

Dixon-Coles cuts log-loss by **~21%** vs uniform and **~18%** vs the base rate, both with bootstrap CIs that exclude zero — i.e. the model has genuine, statistically distinguishable predictive content on real international matches, not just chance variation. ECE of 0.018 means the predicted probabilities are on average within ~1.8 percentage points of empirical frequencies — well-calibrated by typical 1X2-forecasting standards.

What this is *not*: a comparison to a sharp market. International tournament closing odds aren't freely redistributable, so there's no Pinnacle-equivalent reference here. Adding one (Euro 2024 / Copa America 2024 paid feeds) is the obvious next step.

---

## Project Structure
```text
app.py                      # Main Streamlit app (setup → chat → simulation → visuals)
ui.py                       # Styling, setup screen UI, bracket rendering, narration text
mistral_agent.py            # Mistral scenario agent + JSON action parsing + fallback parser
simulation.py               # Dixon-Coles match engine, group/knockout sim, Monte Carlo,
                            #   predict_match_probs (analytic 1X2 / totals / BTTS)
data.py                     # Teams/ratings/form/key players, groups, bracket, H2H
backtest.py                 # Scoring rules (log loss, Brier, RPS, ECE),
                            #   reliability bins, paired bootstrap — pure functions
dc_fit.py                   # MLE fitter for Dixon-Coles parameters (L-BFGS-B),
                            #   with per-match neutral-venue handling
scripts/fit_international.py# Fits team strengths on ~4yr of real international
                            #   matches (martj42/international_results) and writes
                            #   team_ratings.json, consumed by data.py at import time
scripts/validate_international.py
                            # Out-of-sample calibration: train/test split on
                            #   international matches, scores DC vs baselines,
                            #   paired-bootstrap CI on the gap, reliability plot
team_ratings.json           # (gitignored) generated MLE-fit ratings + form
reports/                    # (gitignored) generated intl_metrics.md, plots, JSON
data_raw/                   # (gitignored) cached martj42 results CSV
requirements.txt            # Python dependencies (incl. numpy, scipy, matplotlib)
.env.example                # Environment template (keys)
.gitignore                  # Ignores .env, caches, .streamlit/, reports/, data_raw/
```

---

## Customization

### Change team strengths / form
Edit `data.py`:
- `attack`, `defense`, `midfield` control strength.
- `form` is an extra multiplier used by the match engine.
- `key_players` is used for injury-style prompts (and fallback parsing).

### Update groups or qualification assumptions
- Groups are defined in `GROUPS` in `data.py`.
- The “6 undecided slots” are defined in `PLAYOFF_SLOTS`.

### Change ElevenLabs voice
In `app.py`, update:
- `voice_id="bVM5MBBFUy5Uve0cooHn"`
- `model_id="eleven_multilingual_v2"`

---

## Troubleshooting

**“Set MISTRAL_API_KEY in .env” warning**
- Normal if you didn’t configure Mistral.
- The app still works; scenario chat falls back to keyword parsing.

**Voice is disabled**
- Set `ELEVENLABS_API_KEY` and ensure the `elevenlabs` package is installed (it is included in `requirements.txt`).
- Use the mute button (🔊/🔇) to toggle.

**Flags not showing**
- Flags are rendered via images to avoid emoji rendering issues in browsers.
- Make sure you have internet access if flags are fetched externally.

---

## Roadmap Ideas
- **Market-reference backtest on internationals.** Add Euro 2024 / Copa America 2024 closing odds from a paid feed and run de-vigging + Kelly + scoring rules vs the market. (No free equivalent of football-data.co.uk exists for international tournaments, so this requires either scraping or a paid data subscription.)
- **Walk-forward validation.** Replace the single 80/20 split with rolling-origin evaluation: refit at each tournament window so the test set always reflects the deployment regime (recent international fixture density / opponent quality).
- **Parametric bootstrap on attack/defense.** The fitted ratings are MLE point estimates. Resample matches and refit to get per-team CIs, then propagate that uncertainty into Monte Carlo win-probability bands.
- **MC standard errors.** Tournament win probabilities are reported as point estimates; common-random-numbers and antithetic variates would tighten them, and the binomial SE on `n` simulations should be shown next to each percentage.
- **Hierarchical / shrinkage prior on attack/defense.** Teams with very few internationals (e.g. New Caledonia) get noisy MLEs. A partial-pooling prior would stabilise them.
- **Streamlit page** surfacing the calibration report, fit diagnostics, and per-team match-share counts in-app.

---

## Author
**Håvard Fossdal**  
M.Sc. Industrial Mathematics (Statistics/ML) @ NTNU
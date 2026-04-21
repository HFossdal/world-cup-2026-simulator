# World Cup 2026 Simulator ⚽️
**Probabilistic World Cup forecasting with a Dixon-Coles match engine, a calibration backtest against Pinnacle closing odds, and an AI scenario chat.**

A Streamlit app + backtesting pipeline that:
- simulates the full 48-team tournament (single run + bracket, or Monte Carlo),
- models scorelines with the **Dixon-Coles bivariate distribution** (not independent Poisson),
- ships a **calibration backtest** (`scripts/backtest_report.py`) that fits the DC model by MLE on historical football data and evaluates it on held-out matches using log loss, Brier, RPS, ECE, and fractional-Kelly PnL against sharp-market closing odds,
- exposes an AI scenario chat for "what-if" questions,
- supports optional voice narration via ElevenLabs.

> ⚠️ The tournament sim uses hand-curated international team ratings in `data.py` and is for exploration, not live betting. The calibration backtest is evaluated on club-league data (football-data.co.uk) where high-quality closing odds exist.

---

## Table of Contents
- [Features](#features)
- [Quickstart](#quickstart)
- [API Keys (Optional)](#api-keys-optional)
- [How to Use](#how-to-use)
- [Scenario Examples](#scenario-examples)
- [How Scenarios Work (Under the Hood)](#how-scenarios-work-under-the-hood)
- [How the Simulation Works](#how-the-simulation-works)
- [Calibration Backtest](#calibration-backtest)
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

### 2) Configure environment (optional, but recommended)
```bash
cp .env.example .env
# then edit .env
```

### 3) Run
```bash
streamlit run app.py
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
- **Neutral venue, always.** Every World Cup match is played on neutral ground, so the simulator applies **no home-advantage term** — no `gamma`, no home/away asymmetry. Lambdas are a pure function of attack/defense + form.
- **Expected goals:** team attack / defense / midfield multipliers and a small "form" factor produce `(lambda_a, lambda_b)`; these feed the DC sampler (`simulation.simulate_match`) and the analytic probability function (`simulation.predict_match_probs`).
- **Group standings:** points → goal difference → goals for → FIFA ranking (final tie-breaker).
- **48-team format:** 12 groups, top 2 qualify + **8 best third-place** teams.
- **Knockouts:** no draws — extra time (Dixon-Coles-sampled) + penalties if still level.
- **Head-to-head nudges** are layered on for a handful of classic matchups (see `data.HEAD_TO_HEAD`).

Note: the calibration backtest in `scripts/backtest_report.py` *does* fit a `gamma` home-advantage term, because it runs on club-league data where home/away is a real effect. That `gamma` is scoped to the backtest only — the WC simulator never touches it.

All logic lives in `simulation.py`; format/bracket slots and ratings in `data.py`.

## Calibration Backtest

The file `scripts/backtest_report.py` is the quant-evaluation side of the project. It answers the question *"is this model's probability distribution actually calibrated, and would it make money against a sharp market?"* — the only questions that matter for a probabilistic forecaster.

What it does:
1. Downloads a free CSV from football-data.co.uk (Premier League, Bundesliga, etc.) with match results + Pinnacle closing odds.
2. Splits chronologically into train / test.
3. Fits the full Dixon-Coles model on the train split by **MLE** (L-BFGS-B) — per-team attack / defense, home advantage `gamma`, baseline rate `mu`, and `rho` — with optional Dixon-Coles exponential time-decay.
4. Scores every test-set match with four forecasters:
   - Uniform (1/3, 1/3, 1/3) — information-free baseline
   - Home-advantage base rate (0.46, 0.27, 0.27)
   - **Market** — Pinnacle closing odds after **Shin de-vigging** (sharp-market reference)
   - **Dixon-Coles MLE**
5. Computes **log loss, Brier score, ranked probability score, expected calibration error**, runs a **fractional-Kelly backtest** against Pinnacle, and writes:
   - `reports/metrics.md` — table + commentary
   - `reports/reliability.png` — calibration diagram
   - `reports/kelly_pnl.png` — bankroll curve
   - `reports/metrics.json` — machine-readable

Run it:
```bash
python scripts/backtest_report.py --league E0 --season 2324
# optional: time-decayed likelihood (xi per day, Dixon-Coles used ~0.0065)
python scripts/backtest_report.py --league D1 --season 2223 --xi 0.0065
```

Sample output on the 2023-24 Premier League (114 held-out matches):

| Forecaster | Log loss ↓ | Brier ↓ | RPS ↓ |
|---|---:|---:|---:|
| Uniform | 1.099 | 0.667 | 0.235 |
| Base-rate | 1.057 | 0.637 | 0.224 |
| **Market (Pinnacle, Shin)** | **0.869** | **0.505** | **0.162** |
| Dixon-Coles MLE | 0.937 | 0.554 | 0.183 |

Dixon-Coles beats both naive baselines by a wide margin and lands ~8% short of Pinnacle in log loss. That is the honest result: a vanilla DC fit shouldn't beat the sharpest football market in the world. The Kelly backtest returns negative PnL, as expected when the model is less calibrated than the line — which is exactly what validates the test.

---

## Project Structure
```text
app.py                      # Main Streamlit app (setup → chat → simulation → visuals)
ui.py                       # Styling, setup screen UI, bracket rendering, narration text
mistral_agent.py            # Mistral scenario agent + JSON action parsing + fallback parser
simulation.py               # Dixon-Coles match engine, group/knockout sim, Monte Carlo,
                            #   predict_match_probs (analytic 1X2 / totals / BTTS)
data.py                     # Teams/ratings/form/key players, groups, bracket, H2H
backtest.py                 # De-vigging (proportional, Shin), log loss, Brier, RPS,
                            #   ECE, reliability bins, Kelly backtest — pure functions
dc_fit.py                   # MLE fitter for Dixon-Coles parameters (L-BFGS-B)
scripts/backtest_report.py  # End-to-end: download → fit → evaluate → plot → report
reports/                    # (gitignored) generated metrics.md, plots, JSON
data_raw/                   # (gitignored) downloaded football-data.co.uk CSVs
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
- Multi-season MLE fit with Dixon-Coles time decay (`--xi 0.0065`), bootstrap CIs on the log-loss gap to market
- Closing-line-value (CLV) analysis — did the model's picks move Pinnacle's line in the model's favour?
- Repeat the backtest across D1 / I1 / SP1 / F1 to show the pipeline isn't overfit to one league
- Derive attack/defense ratings for international teams from a historical-results Elo fit (bundled via `martj42/international_results`) to replace the hand-curated multipliers in `data.py`
- Streamlit page that surfaces the calibration report in-app
- Real-time match updates from live APIs; persist scenarios (export/import)

---

## Author
**Håvard Fossdal**  
M.Sc. Industrial Mathematics (Statistics/ML) @ NTNU
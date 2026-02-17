# World Cup 2026 Simulator ⚽️
**AI-powered World Cup simulation with interactive scenarios and optional live voice narration**

A Streamlit app that lets you:
- pick the last undecided qualification slots,
- ask “what-if” questions in natural language (AI-powered when configured),
- simulate a full tournament once (with a full bracket),
- or run Monte Carlo batches to estimate win & stage probabilities.

> ⚠️ This is a *fun simulator*, not an official predictor. Team ratings + form are hand-curated in `data.py`, and match outcomes are sampled (Poisson-based).

---

## Table of Contents
- [Features](#features)
- [Quickstart](#quickstart)
- [API Keys (Optional)](#api-keys-optional)
- [How to Use](#how-to-use)
- [Scenario Examples](#scenario-examples)
- [How Scenarios Work (Under the Hood)](#how-scenarios-work-under-the-hood)
- [How the Simulation Works](#how-the-simulation-works)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Roadmap Ideas](#roadmap-ideas)
- [Author](#author)

---

## Features

### 🧩 Interactive Setup
- The app assumes **42 teams are confirmed + 6 playoff slots** remain.
- You choose the 6 remaining teams from candidate lists (or click **Use Most Likely** / **Randomize**).

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
- **Match engine:** Poisson sampling around an expected goals rate (xG-like), driven by:
  - team attack/defense/midfield multipliers,
  - a small “form” factor,
  - optional head-to-head nudges (for some classic matchups).
- **Group standings:** points → goal difference → goals for → FIFA ranking (as a final tie-breaker).
- **48-team format:** 12 groups, top 2 qualify + **8 best third-place** teams.
- **Knockouts:** no draws — extra time + penalties if needed.

All logic lives in `simulation.py`, and format/bracket slots are defined in `data.py`.

---

## Project Structure
```text
app.py              # Main Streamlit app (setup → chat → simulation → visuals)
ui.py               # Styling, setup screen UI, bracket rendering, narration text
mistral_agent.py    # Mistral scenario agent + JSON action parsing + fallback parser
simulation.py       # Poisson match engine, group/knockout sim, Monte Carlo
data.py             # Teams/ratings/form/key players, groups, bracket slot mapping, H2H
requirements.txt    # Python dependencies
.env.example        # Environment template (keys)
.gitignore          # Ignores .env, caches, .streamlit/
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
- Real-time match updates from live APIs
- Sync bracket highlight with narration (real-time)
- Upload custom rating datasets / presets
- Persist scenarios (export/import) for sharing

---

## Author
**Håvard Fossdal**  
M.Sc. Industrial Mathematics (Statistics/ML) @ NTNU
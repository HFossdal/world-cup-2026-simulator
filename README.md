# ⚽ World Cup 2026 Simulator

A production-quality Streamlit app that simulates the FIFA World Cup 2026 using Poisson-based match modeling, Mistral AI scenario analysis, and ElevenLabs voice narration.

## Features

### Simulation Engine
- **Poisson distribution** match simulation based on team attack/defense/midfield strength ratings
- Full **48-team tournament**: 12 groups of 4, accurate FIFA rules (3pts/1pt/0pt)
- **Best 8 third-placed teams** advance via backtracking bracket assignment
- Knockout bracket: **R32 → R16 → QF → SF → 3rd Place → Final**
- **Extra time and penalty shootouts** for knockout draws
- **Monte Carlo analysis** (100–5,000 simulations) with win probabilities

### SVG Bracket Visualization
- Real tournament bracket tree rendered as SVG
- Rounds displayed as columns: R32 → R16 → QF → SF → Final → Champion
- Connector lines between rounds showing advancement path
- Winners highlighted in gold, champion box at the end
- Horizontally scrollable for any screen size

### Mistral AI Scenario Mode
Chat-driven "what-if" analysis:
- *"What if Norway beats France?"* → Lock result, re-simulate, show bracket
- *"Make Brazil 20% stronger"* → Adjust ratings, re-simulate
- *"What if Mbappé is injured?"* → Reduce France's attack rating
- *"Show me the 10 most likely winners"* → Trigger Monte Carlo
- *"Reset to baseline"* → Undo all modifications

### ElevenLabs Voice Narration
- After each simulation, the AI narrates the champion's path aloud
- Uses ElevenLabs streaming TTS with Rachel voice
- Mute/unmute toggle in the top-right corner
- Gracefully skips if no API key is set

## Quick Start

```bash
pip install -r requirements.txt

# Set your API keys
cp .env.example .env
# Edit .env with your Mistral and (optional) ElevenLabs keys

streamlit run app.py
```

## Project Structure

```
├── app.py              # Main Streamlit app (single-page flow)
├── data.py             # 48 teams, 12 groups, bracket structure, H2H
├── simulation.py       # Poisson engine, group/knockout sim, Monte Carlo
├── mistral_agent.py    # Mistral AI chat agent for scenarios
├── ui.py               # SVG bracket, CSS theme, group tables, narration
├── .env                # API keys (git-ignored)
├── .env.example        # Template
├── requirements.txt    # Dependencies
└── README.md
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MISTRAL_API_KEY` | Yes | Mistral AI API key for scenario chat |
| `ELEVENLABS_API_KEY` | No | ElevenLabs API key for voice narration |

## Data Sources

- Team ratings: Based on FIFA rankings and recent form (early 2026)
- Group draw: Official FIFA World Cup 2026 Final Draw (December 5, 2025)
- Bracket structure: FIFA tournament regulations (R32 seeding chart)
- Head-to-head: Historical international football records

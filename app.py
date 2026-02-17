"""
World Cup 2026 Simulator — Main Streamlit Application

Single-page guided flow: Mistral AI chat at top drives the simulation,
bracket visualization renders below after each run.

Usage:  streamlit run app.py
"""

# TODO: As Mistral speaks about a specific country, highlight that
# country's matches in the bracket in real-time (sync audio + visual)

from __future__ import annotations

import base64
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# --- Streamlit Cloud: mirror secrets into env (so os.getenv keeps working) ---
def _mirror_secrets_to_env(keys: list[str]) -> None:
    for k in keys:
        # st.secrets.get returns None if missing
        v = st.secrets.get(k, None)
        if v and not os.getenv(k):
            os.environ[k] = str(v)

_mirror_secrets_to_env(["MISTRAL_API_KEY", "ELEVENLABS_API_KEY"])

from data import TEAMS, GROUPS, get_teams_copy, get_groups_copy, PLAYOFF_SLOTS
from simulation import (
    simulate_match,
    simulate_tournament,
    run_monte_carlo,
)
from mistral_agent import MistralScenarioAgent, apply_modifications
from ui import (
    inject_css,
    render_header,
    render_bracket,
    render_all_groups,
    render_monte_carlo,
    generate_narration,
    generate_mc_narration,
    render_setup_screen,
    render_scenario_chips,
)

# ---------------------------------------------------------------------------
# ElevenLabs TTS (optional)
# ---------------------------------------------------------------------------
try:
    from elevenlabs import ElevenLabs as ElevenLabsClient

    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False


def speak(text: str) -> bytes | None:
    """Generate TTS audio bytes via ElevenLabs. Returns None on failure."""
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key or not ELEVENLABS_AVAILABLE:
        print("[TTS] Skipped — no API key or elevenlabs not installed")
        return None
    try:
        client = ElevenLabsClient(api_key=api_key)
        print(f"[TTS] Calling ElevenLabs — voice_id=bVM5MBBFUy5Uve0cooHn, text length={len(text)}")
        audio_iter = client.text_to_speech.convert(
            voice_id="bVM5MBBFUy5Uve0cooHn",
            text=text,
            model_id="eleven_multilingual_v2",
        )
        audio_bytes = b"".join(audio_iter)
        print(f"[TTS] Success — {len(audio_bytes)} bytes of audio")
        return audio_bytes
    except Exception as e:
        print(f"[TTS] ERROR: {type(e).__name__}: {e}")
        return None


def play_audio(placeholder, audio_bytes: bytes):
    """Play audio through the single placeholder. Browser stops naturally."""
    b64 = base64.b64encode(audio_bytes).decode()
    placeholder.markdown(
        f'<audio autoplay><source src="data:audio/mpeg;base64,{b64}" '
        f'type="audio/mpeg"></audio>',
        unsafe_allow_html=True,
    )


def extract_focused_team(text: str, teams: dict[str, dict]) -> str | None:
    """Extract a team code from user text if they explicitly mention a country.
    Returns None if no specific team is mentioned."""
    text_lower = text.lower()
    # Check full country names first (longer match = more specific)
    best_match: str | None = None
    best_len = 0
    for code, data in teams.items():
        name = data.get("name", "").lower()
        if name and name in text_lower and len(name) > best_len:
            best_match = code
            best_len = len(name)
    # Also check 3-letter codes (e.g. "FRA", "BRA")
    if not best_match:
        for code in teams:
            if code.lower() in text_lower.split():
                best_match = code
                break
    return best_match


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="World Cup 2026 Simulator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_css()

# Single audio placeholder — the ONLY element that ever holds <audio> HTML.
# Overwritten each time, cleared after playback, so reruns never replay.
audio_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False
if "playoff_selections" not in st.session_state:
    st.session_state.playoff_selections = {}
if "active_groups" not in st.session_state:
    st.session_state.active_groups = None
if "teams" not in st.session_state:
    st.session_state.teams = get_teams_copy()
if "locked_results" not in st.session_state:
    st.session_state.locked_results = {}
if "round_constraints" not in st.session_state:
    st.session_state.round_constraints = {}
if "tournament_result" not in st.session_state:
    st.session_state.tournament_result = None
if "mc_data" not in st.session_state:
    st.session_state.mc_data = None
if "agent" not in st.session_state:
    st.session_state.agent = MistralScenarioAgent()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "change_log" not in st.session_state:
    st.session_state.change_log = []
if "muted" not in st.session_state:
    st.session_state.muted = False
if "highlight_team" not in st.session_state:
    st.session_state.highlight_team = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "last_audio_played" not in st.session_state:
    st.session_state.last_audio_played = None


# ═══════════════════════════════════════════════════════════════════════════
# SETUP SCREEN — shown before simulation begins
# ═══════════════════════════════════════════════════════════════════════════
if not st.session_state.setup_complete:
    start_clicked, selections = render_setup_screen()

    if start_clicked:
        st.session_state.playoff_selections = selections
        st.session_state.active_groups = get_groups_copy(selections)
        st.session_state.teams = get_teams_copy(selections)
        st.session_state.agent = MistralScenarioAgent(
            team_codes=list(st.session_state.teams.keys()),
            groups=st.session_state.active_groups,
        )
        st.session_state.setup_complete = True
        st.rerun()

    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION INTERFACE — shown after setup is complete
# ═══════════════════════════════════════════════════════════════════════════
teams = st.session_state.teams
agent: MistralScenarioAgent = st.session_state.agent
active_groups = st.session_state.active_groups or GROUPS

# ---------------------------------------------------------------------------
# Sidebar — controls & status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")

    if st.button("🏆 Simulate Tournament", use_container_width=True, type="primary"):
        with st.spinner("Simulating..."):
            result = simulate_tournament(teams, active_groups, st.session_state.locked_results, st.session_state.round_constraints)
            st.session_state.tournament_result = result
            st.session_state.mc_data = None
            st.session_state.highlight_team = None
            narration = generate_narration(result, teams)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": narration}
            )
        st.rerun()

    mc_n = st.select_slider("Monte Carlo runs", [100, 500, 1000, 5000], 1000)
    if st.button(f"📊 Run {mc_n} Simulations", use_container_width=True):
        with st.spinner(f"Running {mc_n} sims..."):
            mc_data = run_monte_carlo(teams, mc_n, active_groups, st.session_state.locked_results, st.session_state.round_constraints)
            st.session_state.mc_data = mc_data
            st.session_state.tournament_result = None
            st.session_state.highlight_team = None
            narration = generate_mc_narration(mc_data, teams)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": narration}
            )
        st.rerun()

    st.markdown("---")
    if st.button("🔄 Reset All Modifications", use_container_width=True):
        sels = st.session_state.playoff_selections
        st.session_state.teams = get_teams_copy(sels)
        st.session_state.locked_results = {}
        st.session_state.round_constraints = {}
        st.session_state.change_log = []
        st.session_state.tournament_result = None
        st.session_state.mc_data = None
        st.session_state.highlight_team = None
        agent.reset_conversation()
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "All modifications reset to baseline."}
        )
        teams = st.session_state.teams
        st.rerun()

    if st.button("Change Lineup", use_container_width=True):
        st.session_state.setup_complete = False
        st.session_state.tournament_result = None
        st.session_state.mc_data = None
        st.session_state.chat_history = []
        st.session_state.change_log = []
        st.rerun()

    if st.session_state.change_log:
        st.markdown("### Active Modifications")
        for change in st.session_state.change_log[-10:]:
            st.markdown(f"- {change}")

    # Connection status
    st.markdown("---")
    if agent.is_available:
        st.success("Mistral AI connected", icon="✅")
    else:
        st.warning("Set MISTRAL_API_KEY in .env", icon="⚠️")
    el_key = os.getenv("ELEVENLABS_API_KEY", "")
    if el_key and ELEVENLABS_AVAILABLE:
        st.success("ElevenLabs TTS connected", icon="🔊")
    else:
        st.info("Voice disabled (no ElevenLabs key)", icon="🔇")


# ---------------------------------------------------------------------------
# Main area — Header + Mute toggle
# ---------------------------------------------------------------------------

col_header, col_mute = st.columns([10, 1])
with col_header:
    render_header()
with col_mute:
    st.write("")  # spacer
    mute_label = "🔇" if st.session_state.muted else "🔊"
    if st.button(mute_label, key="mute_toggle", help="Toggle voice narration"):
        st.session_state.muted = not st.session_state.muted
        st.rerun()

# ---------------------------------------------------------------------------
# API key banners (main area)
# ---------------------------------------------------------------------------
if not agent.is_available:
    st.warning(
        "Mistral API key not found. Add `MISTRAL_API_KEY` to your `.env` file "
        "to enable AI-powered scenario analysis. Fallback keyword parsing is active.",
        icon="⚠️",
    )
el_key = os.getenv("ELEVENLABS_API_KEY", "")
if not el_key or not ELEVENLABS_AVAILABLE:
    st.info(
        "Voice narration disabled. Add `ELEVENLABS_API_KEY` to enable commentary.",
        icon="🔇",
    )

# ---------------------------------------------------------------------------
# Chat section
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### 🤖 Scenario Chat")
st.caption(
    "Ask what-if questions, or click an example below to get started:"
)

# Example scenario chips
chip_prompt = render_scenario_chips()
if chip_prompt:
    st.session_state.pending_prompt = chip_prompt
    st.rerun()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Resolve user input: pending chip prompt takes priority over chat_input
user_input = None
if st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
else:
    user_input = st.chat_input("Ask a what-if scenario...")

# Process user input (from chip click or typed message)
if user_input:
    user_input = user_input.strip()
if user_input:
    # New scenario — clear any previous audio and reset tracking
    audio_placeholder.empty()
    st.session_state.last_audio_played = None

    # ── Render user message ──
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    focused = extract_focused_team(user_input, teams)
    if focused:
        st.session_state.highlight_team = focused

    # ── 1. Mistral generates intro ──
    want_voice = not st.session_state.muted
    with st.spinner("Mistral AI is analyzing..."):
        response = agent.chat(user_input, teams)

    # ── 2. Display intro text (no audio) ──
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response.message}
    )
    with st.chat_message("assistant"):
        st.markdown(response.message)

    # Handle reset
    if response.should_reset:
        sels = st.session_state.playoff_selections
        st.session_state.teams = get_teams_copy(sels)
        st.session_state.locked_results = {}
        st.session_state.round_constraints = {}
        st.session_state.change_log = []
        st.session_state.highlight_team = None
        teams = st.session_state.teams
        reset_msg = "All modifications reset to baseline."
        st.session_state.chat_history.append(
            {"role": "assistant", "content": reset_msg}
        )
        with st.chat_message("assistant"):
            st.markdown(reset_msg)

    # Apply modifications
    if response.modifications:
        teams, changes, new_locks, new_constraints = apply_modifications(teams, response.modifications)
        st.session_state.teams = teams
        st.session_state.locked_results.update(new_locks)
        if new_constraints.get("force_exit"):
            existing = st.session_state.round_constraints.get("force_exit", {})
            for rnd, team_set in new_constraints["force_exit"].items():
                if rnd not in existing:
                    existing[rnd] = set()
                existing[rnd].update(team_set)
            st.session_state.round_constraints["force_exit"] = existing
        if new_constraints.get("force_group_winner"):
            existing_gw = st.session_state.round_constraints.get("force_group_winner", set())
            existing_gw.update(new_constraints["force_group_winner"])
            st.session_state.round_constraints["force_group_winner"] = existing_gw
        st.session_state.change_log.extend(changes)
        if changes:
            applied_msg = "**Applied:** " + " | ".join(changes)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": applied_msg}
            )
            with st.chat_message("assistant"):
                st.markdown(applied_msg)

    # ── 3. Run simulation ──
    if response.should_simulate:
        sim_label = (
            f"Running {response.sim_n} simulations..."
            if response.sim_mode == "monte_carlo"
            else "Simulating tournament..."
        )

        with st.spinner(sim_label):
            if response.sim_mode == "monte_carlo":
                sim_result = run_monte_carlo(
                    teams, response.sim_n, active_groups,
                    st.session_state.locked_results,
                    st.session_state.round_constraints,
                )
            else:
                sim_result = simulate_tournament(
                    teams, active_groups,
                    st.session_state.locked_results,
                    st.session_state.round_constraints,
                )

        # ── 4. Generate results narration + audio (nothing shown yet) ──
        if response.sim_mode == "monte_carlo":
            st.session_state.mc_data = sim_result
            st.session_state.tournament_result = None
            narration = generate_mc_narration(sim_result, teams)
        else:
            st.session_state.tournament_result = sim_result
            narration = generate_narration(sim_result, teams)

        # ── 4b. Display results text immediately ──
        st.session_state.chat_history.append(
            {"role": "assistant", "content": narration}
        )
        with st.chat_message("assistant"):
            st.markdown(narration)

        # ── 5. Generate results audio + play (intro already finished) ──
        if want_voice and narration and st.session_state.last_audio_played != "results":
            results_audio = None
            with st.spinner("Generating commentary..."):
                results_audio = speak(narration)
            if results_audio:
                play_audio(audio_placeholder, results_audio)
                st.session_state.last_audio_played = "results"


# ---------------------------------------------------------------------------
# Bracket visualization (appears after simulation)
# ---------------------------------------------------------------------------

if st.session_state.tournament_result:
    st.markdown("---")
    st.markdown("### 🏆 Tournament Bracket")
    render_bracket(
        st.session_state.tournament_result,
        teams,
        highlight_team=st.session_state.highlight_team,
    )

    # Group tables in expander
    with st.expander("📋 Group Stage Tables", expanded=False):
        render_all_groups(st.session_state.tournament_result.group_tables, teams)

        # Best third-placed teams
        third = st.session_state.tournament_result.third_place_ranking
        if third:
            st.markdown("#### Best Third-Placed Teams")
            for i, s in enumerate(third):
                t = teams.get(s.team, {})
                icon = "✅" if i < 8 else "❌"
                st.markdown(
                    f"{icon} {t.get('name', s.team)} — "
                    f"{s.points}pts, GD {s.gd:+d}"
                )

# Monte Carlo results
if st.session_state.mc_data:
    st.markdown("---")
    render_monte_carlo(st.session_state.mc_data, teams)

# Empty state
if not st.session_state.tournament_result and not st.session_state.mc_data:
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;padding:3rem;color:#cccccc;">'
        "<h3>Type a scenario above or click Simulate in the sidebar to begin</h3>"
        "<p>The full tournament bracket will appear here after simulation.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

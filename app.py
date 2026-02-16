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

from data import TEAMS, GROUPS, get_teams_copy
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
        return None
    try:
        client = ElevenLabsClient(api_key=api_key)
        audio_iter = client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
            text=text,
            model_id="eleven_multilingual_v2",
        )
        return b"".join(audio_iter)
    except Exception:
        return None


def autoplay_audio(audio_bytes: bytes):
    """Embed an autoplay HTML audio element in Streamlit."""
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio autoplay><source src="data:audio/mpeg;base64,{b64}" '
        f'type="audio/mpeg"></audio>',
        unsafe_allow_html=True,
    )


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

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "teams" not in st.session_state:
    st.session_state.teams = get_teams_copy()
if "locked_results" not in st.session_state:
    st.session_state.locked_results = {}
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
if "pending_audio" not in st.session_state:
    st.session_state.pending_audio = None

teams = st.session_state.teams
agent: MistralScenarioAgent = st.session_state.agent

# ---------------------------------------------------------------------------
# Sidebar — controls & status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")

    if st.button("🏆 Simulate Tournament", use_container_width=True, type="primary"):
        with st.spinner("Simulating..."):
            result = simulate_tournament(teams, GROUPS, st.session_state.locked_results)
            st.session_state.tournament_result = result
            st.session_state.mc_data = None
            narration = generate_narration(result, teams)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": narration}
            )
            if not st.session_state.muted:
                st.session_state.pending_audio = narration
        st.rerun()

    mc_n = st.select_slider("Monte Carlo runs", [100, 500, 1000, 5000], 1000)
    if st.button(f"📊 Run {mc_n} Simulations", use_container_width=True):
        with st.spinner(f"Running {mc_n} sims..."):
            mc_data = run_monte_carlo(teams, mc_n, GROUPS, st.session_state.locked_results)
            st.session_state.mc_data = mc_data
            top5 = sorted(mc_data["win_probs"].items(), key=lambda x: x[1], reverse=True)[:5]
            summary = "**Top 5 winners:** " + ", ".join(
                f"{teams[c]['flag']} {teams[c]['name']} ({p:.1f}%)" for c, p in top5
            )
            st.session_state.chat_history.append({"role": "assistant", "content": summary})
        st.rerun()

    st.markdown("---")
    if st.button("🔄 Reset All Modifications", use_container_width=True):
        st.session_state.teams = get_teams_copy()
        st.session_state.locked_results = {}
        st.session_state.change_log = []
        st.session_state.tournament_result = None
        st.session_state.mc_data = None
        agent.reset_conversation()
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "All modifications reset to baseline."}
        )
        teams = st.session_state.teams
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
# Chat section
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### 🤖 Scenario Chat")
st.caption(
    "Ask what-if questions: *\"What if Norway beats France?\"* · "
    "*\"Make Brazil 20% stronger\"* · *\"Show me the most likely winners\"* · "
    "*\"Reset to baseline\"*"
)

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Play any pending audio (from previous run)
if st.session_state.pending_audio and not st.session_state.muted:
    audio_bytes = speak(st.session_state.pending_audio)
    if audio_bytes:
        autoplay_audio(audio_bytes)
    st.session_state.pending_audio = None

# Chat input
if user_input := st.chat_input("Ask a what-if scenario..."):
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Get AI response
    response = agent.chat(user_input, teams)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response.message}
    )

    # Handle reset
    if response.should_reset:
        st.session_state.teams = get_teams_copy()
        st.session_state.locked_results = {}
        st.session_state.change_log = []
        teams = st.session_state.teams
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "All modifications reset to baseline."}
        )

    # Apply modifications
    if response.modifications:
        teams, changes, new_locks = apply_modifications(teams, response.modifications)
        st.session_state.teams = teams
        st.session_state.locked_results.update(new_locks)
        st.session_state.change_log.extend(changes)
        if changes:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "**Applied:** " + " | ".join(changes)}
            )

    # Run simulation if requested
    if response.should_simulate:
        if response.sim_mode == "monte_carlo":
            with st.spinner(f"Running {response.sim_n} simulations..."):
                mc_data = run_monte_carlo(
                    teams, response.sim_n, GROUPS, st.session_state.locked_results
                )
                st.session_state.mc_data = mc_data
                top5 = sorted(
                    mc_data["win_probs"].items(), key=lambda x: x[1], reverse=True
                )[:5]
                summary = "**Top 5 winners:** " + ", ".join(
                    f"{teams[c]['flag']} {teams[c]['name']} ({p:.1f}%)"
                    for c, p in top5
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": summary}
                )
        else:
            with st.spinner("Simulating tournament..."):
                result = simulate_tournament(
                    teams, GROUPS, st.session_state.locked_results
                )
                st.session_state.tournament_result = result
                narration = generate_narration(result, teams)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": narration}
                )
                if not st.session_state.muted:
                    st.session_state.pending_audio = narration

    st.rerun()


# ---------------------------------------------------------------------------
# Bracket visualization (appears after simulation)
# ---------------------------------------------------------------------------

if st.session_state.tournament_result:
    st.markdown("---")
    st.markdown("### 🏆 Tournament Bracket")
    render_bracket(st.session_state.tournament_result, teams)

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
                    f"{icon} {t.get('flag', '')} {t.get('name', s.team)} — "
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

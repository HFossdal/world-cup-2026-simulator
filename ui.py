"""
World Cup 2026 Simulator — UI Components

SVG bracket visualization, custom CSS theme, group tables,
match cards, and tournament narration.
"""

from __future__ import annotations

import html as html_mod
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from data import TEAMS, GROUPS, CONFIRMED_QUALIFIED, PLAYOFF_SLOTS, get_h2h
from simulation import MatchResult, GroupStanding, TournamentResult

# ---------------------------------------------------------------------------
# CSS theme — WCAG AA compliant contrast
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ── Base dark-green theme ────────────────────────────────────── */
.stApp {
    background-color: #0d1f0d !important;
    color: #ffffff !important;
}

/* Force white text everywhere */
.stApp, .stApp p, .stApp span, .stApp li, .stApp label,
.stApp div, .stApp td, .stApp th,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span {
    color: #ffffff !important;
}

/* Gold headings */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #FFD700 !important;
}

/* Secondary / muted text */
.stApp .secondary, .stCaption, small {
    color: #cccccc !important;
}

/* ── Header ───────────────────────────────────────────────────── */
.sim-header {
    text-align: center;
    padding: 1.2rem 0 0.6rem 0;
}
.sim-header h1 {
    color: #FFD700 !important;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 0;
}
.sim-header p {
    color: #cccccc !important;
    font-size: 1rem;
    margin-top: 0.2rem;
}

/* ── Chat messages ────────────────────────────────────────────── */
div[data-testid="stChatMessage"] {
    background: #132613 !important;
    border: 1px solid #2d5a2d !important;
    border-radius: 8px !important;
}

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton > button {
    background-color: #1a3a1a !important;
    color: #FFD700 !important;
    border: 1px solid #FFD700 !important;
}
.stButton > button:hover {
    background-color: #2d5a2d !important;
}

/* ── Group table header ───────────────────────────────────────── */
.group-header {
    color: #FFD700 !important;
    font-size: 1.2rem;
    font-weight: 700;
    border-bottom: 2px solid #2d5a2d;
    padding-bottom: 0.3rem;
    margin-bottom: 0.5rem;
}

/* ── Metrics ──────────────────────────────────────────────────── */
div[data-testid="stMetricValue"] {
    color: #FFD700 !important;
}
div[data-testid="stMetricLabel"] {
    color: #cccccc !important;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #0a180a !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label {
    color: #ffffff !important;
}

/* ── Dataframe overrides ──────────────────────────────────────── */
.stDataFrame {
    background-color: #0d1f0d !important;
}

/* ── Chat input ───────────────────────────────────────────────── */
div[data-testid="stChatInput"] {
    border-color: #4CAF50 !important;
}
div[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    background-color: #1a2e1a !important;
    border: 2px solid #4CAF50 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    font-size: 16px !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
    color: #888888 !important;
    opacity: 1 !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #FFD700 !important;
    box-shadow: 0 0 8px rgba(255,215,0,0.3) !important;
    outline: none !important;
}
/* Streamlit wraps chat input in a container with bottom border */
div[data-testid="stChatInput"] > div {
    border-color: #4CAF50 !important;
}
div[data-testid="stBottom"] {
    background-color: #0d1f0d !important;
}
div[data-testid="stBottom"] > div {
    background-color: #0d1f0d !important;
}

/* ── Setup screen ────────────────────────────────────────────── */
.setup-header {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
}
.setup-header h1 {
    color: #FFD700 !important;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 0.2rem;
}
.setup-header p {
    color: #cccccc !important;
    font-size: 1rem;
}
.setup-team-confirmed {
    background: #132613;
    border-left: 3px solid #4CAF50;
    padding: 4px 10px;
    margin: 2px 0;
    border-radius: 0 4px 4px 0;
    color: #ffffff;
    font-size: 0.9rem;
}
.setup-slot-undecided {
    background: #1a2e1a;
    border: 1px dashed #FFD700;
    padding: 6px 10px;
    margin: 4px 0;
    border-radius: 4px;
    color: #FFD700;
    font-size: 0.9rem;
}
</style>
"""


def inject_css():
    """Inject custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Setup screen
# ---------------------------------------------------------------------------

def render_setup_screen():
    """Render the tournament setup screen where users pick the 6 undecided
    playoff slots. Returns (start_clicked, selections_dict) each call."""
    st.markdown(
        '<div class="setup-header">'
        "<h1>FIFA WORLD CUP 2026 DRAW</h1>"
        "<p>42 teams confirmed &mdash; 6 playoff spots to be decided (March 2026)</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Build a lookup: group_letter → list of slot dicts for that group
    slots_by_group: dict[str, list[dict]] = {}
    for slot in PLAYOFF_SLOTS:
        slots_by_group.setdefault(slot["group"], []).append(slot)

    # The position index in GROUPS that each slot occupies
    # (this is needed to know which row in the group is undecided)
    slot_pos_map: dict[str, dict[int, dict]] = {}  # group → {pos_idx: slot}
    _pos_lookup = {
        "slot_A3": ("A", 3), "slot_B3": ("B", 3), "slot_D3": ("D", 3),
        "slot_F2": ("F", 2), "slot_I3": ("I", 3), "slot_K3": ("K", 3),
    }
    for slot in PLAYOFF_SLOTS:
        grp, pos = _pos_lookup[slot["id"]]
        slot_pos_map.setdefault(grp, {})[pos] = slot

    selections: dict[str, str] = {}

    # Display 12 groups in a 3-column grid
    group_letters = list(GROUPS.keys())
    for row_start in range(0, 12, 3):
        cols = st.columns(3)
        for col_idx, col in enumerate(cols):
            grp_idx = row_start + col_idx
            if grp_idx >= len(group_letters):
                break
            letter = group_letters[grp_idx]
            team_codes = GROUPS[letter]
            with col:
                st.markdown(f'<div class="group-header">Group {letter}</div>',
                            unsafe_allow_html=True)
                for pos_idx, code in enumerate(team_codes):
                    # Check if this position is an undecided slot
                    slot_info = slot_pos_map.get(letter, {}).get(pos_idx)
                    if slot_info:
                        # Undecided slot — show selectbox
                        candidates = slot_info["candidates"]
                        candidate_labels = [
                            f"{TEAMS[c]['flag']} {TEAMS[c]['name']}"
                            for c in candidates
                        ]
                        st.markdown(
                            f'<div class="setup-slot-undecided">'
                            f'Pos {pos_idx + 1} — {slot_info["label"]}</div>',
                            unsafe_allow_html=True,
                        )
                        chosen_idx = st.selectbox(
                            f"Pick team for {slot_info['label']}",
                            range(len(candidates)),
                            format_func=lambda i, cl=candidate_labels: cl[i],
                            key=f"setup_{slot_info['id']}",
                            label_visibility="collapsed",
                        )
                        selections[slot_info["id"]] = candidates[chosen_idx]
                    else:
                        # Confirmed team — locked display
                        t = TEAMS.get(code, {})
                        st.markdown(
                            f'<div class="setup-team-confirmed">'
                            f'{t.get("flag", "")} {t.get("name", code)}</div>',
                            unsafe_allow_html=True,
                        )

    st.markdown("---")

    # Action buttons
    btn_cols = st.columns(3)
    with btn_cols[0]:
        if st.button("Use Most Likely", use_container_width=True):
            for slot in PLAYOFF_SLOTS:
                selections[slot["id"]] = slot["most_likely"]
                st.session_state[f"setup_{slot['id']}"] = 0  # first = most likely
            st.rerun()
    with btn_cols[1]:
        import random as _rand
        if st.button("Randomize", use_container_width=True):
            for slot in PLAYOFF_SLOTS:
                idx = _rand.randrange(len(slot["candidates"]))
                selections[slot["id"]] = slot["candidates"][idx]
                st.session_state[f"setup_{slot['id']}"] = idx
            st.rerun()

    # Ensure all slots have a selection
    for slot in PLAYOFF_SLOTS:
        if slot["id"] not in selections:
            selections[slot["id"]] = slot["most_likely"]

    with btn_cols[2]:
        start_clicked = st.button(
            "Start Simulating", use_container_width=True, type="primary"
        )

    return start_clicked, selections


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header():
    """Render the app header with optional mute toggle."""
    st.markdown(
        """
        <div class="sim-header">
            <h1>⚽ WORLD CUP 2026 SIMULATOR</h1>
            <p>Powered by Mistral AI &amp; ElevenLabs</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# SVG bracket helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Escape text for safe SVG embedding."""
    return html_mod.escape(str(text))


def _truncate(text: str, max_len: int = 16) -> str:
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _score_label(m: MatchResult) -> tuple[str, str]:
    """Return (score_a_str, score_b_str) including ET/pen info."""
    total_a = m.score_a + m.et_score_a
    total_b = m.score_b + m.et_score_b
    suffix = ""
    if m.penalties:
        suffix = f" ({m.penalty_score_a}-{m.penalty_score_b}p)"
    elif m.extra_time:
        suffix = " aet"
    return str(total_a), str(total_b) + suffix


# ---------------------------------------------------------------------------
# SVG bracket generation
# ---------------------------------------------------------------------------

MATCH_W = 170
MATCH_H = 46
ROW_H = 23
GAP_R32 = 6
ROUND_GAP = 48
PADDING = 12
FONT = "Segoe UI, Arial, Helvetica, sans-serif"

# Colours
COL_BG = "#0d1f0d"
COL_NODE_BG = "#132613"
COL_NODE_BORDER = "#2d5a2d"
COL_WINNER = "#FFD700"
COL_LOSER = "#888888"
COL_TEXT = "#ffffff"
COL_LINE = "#2d5a2d"
COL_LABEL = "#cccccc"


def _draw_match_node(
    parts: list[str],
    m: MatchResult | None,
    x: float,
    y: float,
    teams: dict,
    label: str = "",
    highlight_team: str | None = None,
):
    """Append SVG elements for one match node at (x, y).

    highlight_team: If set to a team code, that team's name is rendered
    in gold/bold. All other teams use neutral white text. When None,
    the bracket is fully neutral (no highlighting).
    """
    # Background rect
    parts.append(
        f'<rect x="{x}" y="{y}" width="{MATCH_W}" height="{MATCH_H}" '
        f'rx="4" fill="{COL_NODE_BG}" stroke="{COL_NODE_BORDER}" stroke-width="1"/>'
    )
    # Divider line
    parts.append(
        f'<line x1="{x}" y1="{y + ROW_H}" x2="{x + MATCH_W}" y2="{y + ROW_H}" '
        f'stroke="{COL_NODE_BORDER}" stroke-width="0.5"/>'
    )

    if m is None:
        # Empty / TBD node
        parts.append(
            f'<text x="{x + MATCH_W / 2}" y="{y + MATCH_H / 2 + 4}" '
            f'text-anchor="middle" fill="{COL_LOSER}" font-size="10" '
            f'font-family="{FONT}">{_esc(label) or "TBD"}</text>'
        )
        return

    ta = teams.get(m.team_a, {})
    tb = teams.get(m.team_b, {})
    name_a = _truncate(ta.get("name", m.team_a))
    name_b = _truncate(tb.get("name", m.team_b))
    score_a, score_b = _score_label(m)

    # Highlighting: only highlight if this specific team is focused
    is_a_highlighted = highlight_team is not None and m.team_a == highlight_team
    is_b_highlighted = highlight_team is not None and m.team_b == highlight_team
    col_a = COL_WINNER if is_a_highlighted else COL_TEXT
    col_b = COL_WINNER if is_b_highlighted else COL_TEXT
    weight_a = "bold" if is_a_highlighted else "normal"
    weight_b = "bold" if is_b_highlighted else "normal"

    # Row A — team name
    parts.append(
        f'<text x="{x + 6}" y="{y + 16}" fill="{col_a}" font-size="11" '
        f'font-weight="{weight_a}" font-family="{FONT}">{_esc(name_a)}</text>'
    )
    # Row A — score
    parts.append(
        f'<text x="{x + MATCH_W - 6}" y="{y + 16}" text-anchor="end" '
        f'fill="{col_a}" font-size="11" font-weight="{weight_a}" '
        f'font-family="{FONT}">{_esc(score_a)}</text>'
    )
    # Row B — team name
    parts.append(
        f'<text x="{x + 6}" y="{y + ROW_H + 16}" fill="{col_b}" font-size="11" '
        f'font-weight="{weight_b}" font-family="{FONT}">{_esc(name_b)}</text>'
    )
    # Row B — score
    parts.append(
        f'<text x="{x + MATCH_W - 6}" y="{y + ROW_H + 16}" text-anchor="end" '
        f'fill="{col_b}" font-size="11" font-weight="{weight_b}" '
        f'font-family="{FONT}">{_esc(score_b)}</text>'
    )


def generate_bracket_svg(
    result: TournamentResult,
    teams: dict,
    highlight_team: str | None = None,
) -> str:
    """Generate a full SVG tournament bracket.

    highlight_team: If set, highlight this team's appearances in gold.
    When None, the bracket is fully neutral.
    """

    rounds_data: list[tuple[str, list[MatchResult | None]]] = [
        ("ROUND OF 32", result.r32_matches),
        ("ROUND OF 16", result.r16_matches),
        ("QUARTERFINALS", result.qf_matches),
        ("SEMIFINALS", result.sf_matches),
        ("FINAL", [result.final_match] if result.final_match else []),
    ]

    # Pad rounds with None to expected sizes so layout works even if
    # some matches are missing
    expected_counts = [16, 8, 4, 2, 1]
    for i, ((name, matches), expected) in enumerate(
        zip(rounds_data, expected_counts)
    ):
        while len(matches) < expected:
            matches.append(None)

    # Calculate vertical positions for each round
    n_r32 = expected_counts[0]
    total_h = n_r32 * MATCH_H + (n_r32 - 1) * GAP_R32 + PADDING * 2
    round_label_h = 24  # Height reserved for round labels at top

    positions: list[list[float]] = []
    for round_idx, (_, matches) in enumerate(rounds_data):
        n = len(matches)
        if round_idx == 0:
            # R32: evenly spaced from top
            ys = [
                PADDING + round_label_h + i * (MATCH_H + GAP_R32)
                for i in range(n)
            ]
        else:
            # Later rounds: centered between feeder match pair
            prev = positions[round_idx - 1]
            ys = []
            for i in range(n):
                idx_a = i * 2
                idx_b = i * 2 + 1
                if idx_a < len(prev) and idx_b < len(prev):
                    center_a = prev[idx_a] + MATCH_H / 2
                    center_b = prev[idx_b] + MATCH_H / 2
                    ys.append((center_a + center_b) / 2 - MATCH_H / 2)
                elif idx_a < len(prev):
                    ys.append(prev[idx_a])
                else:
                    ys.append(total_h / 2 - MATCH_H / 2)
        positions.append(ys)

    n_rounds = len(rounds_data)
    champion_w = 100
    total_w = (
        PADDING * 2
        + n_rounds * MATCH_W
        + (n_rounds - 1) * ROUND_GAP
        + ROUND_GAP
        + champion_w
    )
    total_h_final = total_h + round_label_h

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_w} {total_h_final}" '
        f'style="width:100%;height:auto;display:block;">'
    ]

    # Background
    svg.append(
        f'<rect width="{total_w}" height="{total_h_final}" fill="{COL_BG}"/>'
    )

    # ── Connector lines (drawn first, behind nodes) ──────────────
    for round_idx in range(n_rounds - 1):
        curr_ys = positions[round_idx]
        next_ys = positions[round_idx + 1] if round_idx + 1 < len(positions) else []
        curr_right_x = PADDING + round_idx * (MATCH_W + ROUND_GAP) + MATCH_W
        next_left_x = curr_right_x + ROUND_GAP
        mid_x = (curr_right_x + next_left_x) / 2

        for i in range(0, len(curr_ys), 2):
            if i + 1 < len(curr_ys) and i // 2 < len(next_ys):
                y1 = curr_ys[i] + MATCH_H / 2
                y2 = curr_ys[i + 1] + MATCH_H / 2
                y_next = next_ys[i // 2] + MATCH_H / 2

                # Horizontal from match i → mid
                svg.append(
                    f'<line x1="{curr_right_x}" y1="{y1}" '
                    f'x2="{mid_x}" y2="{y1}" '
                    f'stroke="{COL_LINE}" stroke-width="1.5"/>'
                )
                # Horizontal from match i+1 → mid
                svg.append(
                    f'<line x1="{curr_right_x}" y1="{y2}" '
                    f'x2="{mid_x}" y2="{y2}" '
                    f'stroke="{COL_LINE}" stroke-width="1.5"/>'
                )
                # Vertical connecting the two
                svg.append(
                    f'<line x1="{mid_x}" y1="{y1}" '
                    f'x2="{mid_x}" y2="{y2}" '
                    f'stroke="{COL_LINE}" stroke-width="1.5"/>'
                )
                # Horizontal from vertical midpoint → next match
                svg.append(
                    f'<line x1="{mid_x}" y1="{y_next}" '
                    f'x2="{next_left_x}" y2="{y_next}" '
                    f'stroke="{COL_LINE}" stroke-width="1.5"/>'
                )

    # ── Round labels ─────────────────────────────────────────────
    for round_idx, (round_name, _) in enumerate(rounds_data):
        x = PADDING + round_idx * (MATCH_W + ROUND_GAP) + MATCH_W / 2
        svg.append(
            f'<text x="{x}" y="{PADDING + 14}" text-anchor="middle" '
            f'fill="{COL_LABEL}" font-size="10" font-weight="bold" '
            f'font-family="{FONT}">{_esc(round_name)}</text>'
        )

    # ── Match nodes ──────────────────────────────────────────────
    for round_idx, (round_name, matches) in enumerate(rounds_data):
        x = PADDING + round_idx * (MATCH_W + ROUND_GAP)
        ys = positions[round_idx]
        for i, y in enumerate(ys):
            m = matches[i] if i < len(matches) else None
            _draw_match_node(svg, m, x, y, teams, highlight_team=highlight_team)

    # ── Champion box ─────────────────────────────────────────────
    if result.champion:
        champ = teams.get(result.champion, {})
        champ_x = PADDING + n_rounds * (MATCH_W + ROUND_GAP) - ROUND_GAP + 20
        final_ys = positions[-1]
        champ_y = final_ys[0] + MATCH_H / 2 - 28 if final_ys else total_h_final / 2 - 28

        # Gold-bordered champion box
        svg.append(
            f'<rect x="{champ_x}" y="{champ_y}" width="{champion_w - 10}" '
            f'height="56" rx="6" fill="{COL_NODE_BG}" '
            f'stroke="{COL_WINNER}" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{champ_x + (champion_w - 10) / 2}" y="{champ_y + 22}" '
            f'text-anchor="middle" fill="{COL_WINNER}" font-size="18" '
            f'font-family="{FONT}">&#127942;</text>'
        )
        svg.append(
            f'<text x="{champ_x + (champion_w - 10) / 2}" y="{champ_y + 42}" '
            f'text-anchor="middle" fill="{COL_WINNER}" font-size="11" '
            f'font-weight="bold" font-family="{FONT}">'
            f'{_esc(champ.get("name", result.champion))}</text>'
        )

        # Connector line from final → champion
        if final_ys:
            final_right = PADDING + (n_rounds - 1) * (MATCH_W + ROUND_GAP) + MATCH_W
            svg.append(
                f'<line x1="{final_right}" y1="{final_ys[0] + MATCH_H / 2}" '
                f'x2="{champ_x}" y2="{champ_y + 28}" '
                f'stroke="{COL_WINNER}" stroke-width="2"/>'
            )

    svg.append("</svg>")
    return "\n".join(svg)


def render_bracket(
    result: TournamentResult,
    teams: dict[str, dict],
    highlight_team: str | None = None,
):
    """Render the full knockout bracket as an interactive SVG."""
    svg_html = generate_bracket_svg(result, teams, highlight_team=highlight_team)

    # Wrap in scrollable container
    wrapper = f"""
    <div style="overflow-x:auto; overflow-y:hidden; background:{COL_BG};
                border:1px solid {COL_NODE_BORDER}; border-radius:8px;
                padding:8px;">
        {svg_html}
    </div>
    """

    # Calculate height from number of R32 matches
    n_r32 = 16
    svg_h = n_r32 * MATCH_H + (n_r32 - 1) * GAP_R32 + PADDING * 2 + 40
    components.html(wrapper, height=svg_h + 30, scrolling=True)

    # Third-place match below bracket
    if result.third_place_match:
        m = result.third_place_match
        ta = teams.get(m.team_a, {})
        tb = teams.get(m.team_b, {})
        total_a = m.score_a + m.et_score_a
        total_b = m.score_b + m.et_score_b
        extra = ""
        if m.penalties:
            extra = f" (pens {m.penalty_score_a}-{m.penalty_score_b})"
        elif m.extra_time:
            extra = " AET"
        bold_a = "**" if m.winner == m.team_a else ""
        bold_b = "**" if m.winner == m.team_b else ""
        st.markdown(
            f"**Third-Place Match:** {ta.get('flag', '')} {bold_a}{ta.get('name', '')}{bold_a}"
            f" `{total_a}-{total_b}` "
            f"{bold_b}{tb.get('name', '')}{bold_b} {tb.get('flag', '')}{extra}"
        )


# ---------------------------------------------------------------------------
# Group tables
# ---------------------------------------------------------------------------

def render_group_table(
    group_letter: str,
    standings: list[GroupStanding],
    teams: dict[str, dict],
):
    """Render a single group table."""
    st.markdown(
        f'<div class="group-header">Group {group_letter}</div>',
        unsafe_allow_html=True,
    )
    rows = []
    for i, s in enumerate(standings):
        t = teams.get(s.team, {})
        qualifier = " ✅" if i < 2 else ""
        rows.append({
            "#": i + 1,
            "Team": f"{t.get('flag', '')} {t.get('name', s.team)}{qualifier}",
            "P": s.played, "W": s.wins, "D": s.draws, "L": s.losses,
            "GF": s.gf, "GA": s.ga, "GD": s.gd, "Pts": s.points,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)


def render_all_groups(
    tables: dict[str, list[GroupStanding]],
    teams: dict[str, dict],
):
    """Render all 12 group tables in a 3-column grid."""
    for row_start in range(0, 12, 3):
        cols = st.columns(3)
        for col_idx, col in enumerate(cols):
            grp_idx = row_start + col_idx
            letter = chr(ord("A") + grp_idx)
            if letter in tables:
                with col:
                    render_group_table(letter, tables[letter], teams)


# ---------------------------------------------------------------------------
# Monte Carlo results
# ---------------------------------------------------------------------------

def render_monte_carlo(mc_data: dict, teams: dict[str, dict]):
    """Render Monte Carlo simulation results."""
    win_probs = mc_data["win_probs"]
    stage_probs = mc_data["stage_probs"]
    n = mc_data["n"]

    st.markdown(f"### Monte Carlo Results ({n:,} simulations)")

    top10 = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)[:10]
    chart_data = pd.DataFrame({
        "Team": [f"{teams[c]['flag']} {teams[c]['name']}" for c, _ in top10],
        "Win %": [p for _, p in top10],
    })
    st.bar_chart(chart_data, x="Team", y="Win %", color="#FFD700")

    if mc_data["most_likely_final"]:
        pair = mc_data["most_likely_final"]
        ta = teams.get(pair[0], {})
        tb = teams.get(pair[1], {})
        st.markdown(
            f"**Most Likely Final:** {ta.get('flag', '')} {ta.get('name', pair[0])} "
            f"vs {tb.get('name', pair[1])} {tb.get('flag', '')} "
            f"({mc_data['most_likely_final_pct']:.1f}%)"
        )

    st.markdown("#### Stage Probabilities")
    rows = []
    stages = ["Winner", "Final", "SF", "QF", "R16", "R32", "Group Exit"]
    sorted_teams = sorted(
        stage_probs.keys(),
        key=lambda c: stage_probs[c].get("Winner", 0),
        reverse=True,
    )
    for code in sorted_teams:
        t = teams.get(code, {})
        row = {"Team": f"{t.get('flag', '')} {t.get('name', code)}"}
        for s in stages:
            row[s] = f"{stage_probs[code].get(s, 0):.1f}%"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=600)


# ---------------------------------------------------------------------------
# Tournament narration
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd'][min(n % 10, 4)] if n % 10 < 4 else 'th'}"


def _knockout_drama(m, champ_code: str, teams: dict) -> str:
    """Describe a knockout match in commentator style."""
    opp_code = m.team_b if m.team_a == champ_code else m.team_a
    opp = teams.get(opp_code, {})
    opp_name = opp.get("name", opp_code)
    total_a = m.score_a + m.et_score_a
    total_b = m.score_b + m.et_score_b
    if m.team_a == champ_code:
        score = f"{total_a}-{total_b}"
    else:
        score = f"{total_b}-{total_a}"
    if m.penalties:
        pa = m.penalty_score_a if m.team_a == champ_code else m.penalty_score_b
        pb = m.penalty_score_b if m.team_a == champ_code else m.penalty_score_a
        return f"edged past {opp_name} {score} in a nail-biting penalty shootout, {pa}-{pb}"
    elif m.extra_time:
        return f"battled through extra time to overcome {opp_name} {score}"
    elif (total_a - total_b >= 3) if m.team_a == champ_code else (total_b - total_a >= 3):
        return f"demolished {opp_name} {score} in a dominant display"
    else:
        return f"dispatched {opp_name} {score}"


def generate_narration(result: TournamentResult, teams: dict[str, dict]) -> str:
    """Generate a sports-commentator-style narration of the tournament outcome,
    tracing the champion's path from group stage to final."""
    if not result.champion:
        return "What a tournament! But we couldn't determine a champion this time around."

    champ_code = result.champion
    champ = teams.get(champ_code, {})
    champ_name = champ.get("name", champ_code)
    parts: list[str] = []

    parts.append(
        f"AND THERE IT IS! {champ_name} have done it! "
        f"They are the 2026 FIFA World Cup Champions!"
    )

    # Group stage
    for group, table in result.group_tables.items():
        for i, standing in enumerate(table):
            if standing.team == champ_code:
                if i == 0:
                    parts.append(
                        f"They stormed through Group {group}, "
                        f"finishing top with {standing.points} points "
                        f"and {standing.gf} goals to their name."
                    )
                elif i == 1:
                    parts.append(
                        f"They qualified as runners-up from Group {group} "
                        f"with {standing.points} points, "
                        f"but don't let that fool you — they were saving their best."
                    )
                else:
                    parts.append(
                        f"Remarkably, they scraped through as one of the best "
                        f"third-placed teams from Group {group} with just "
                        f"{standing.points} points. Nobody saw this coming!"
                    )
                break

    # Knockout path
    round_labels = [
        ("Round of 32", result.r32_matches),
        ("Round of 16", result.r16_matches),
        ("Quarterfinals", result.qf_matches),
        ("Semifinals", result.sf_matches),
    ]

    ko_parts: list[str] = []
    for round_name, matches in round_labels:
        for m in matches:
            if m and m.winner == champ_code:
                ko_parts.append(f"In the {round_name}, they {_knockout_drama(m, champ_code, teams)}.")
                break

    if ko_parts:
        parts.append("Then the knockout stage — and what a ride it was!")
        parts.extend(ko_parts)

    # Final
    if result.final_match:
        m = result.final_match
        opp_code = m.team_b if m.team_a == champ_code else m.team_a
        opp = teams.get(opp_code, {})
        opp_name = opp.get("name", opp_code)
        total_a = m.score_a + m.et_score_a
        total_b = m.score_b + m.et_score_b
        if m.team_a == champ_code:
            score = f"{total_a}-{total_b}"
        else:
            score = f"{total_b}-{total_a}"
        extra = ""
        if m.penalties:
            pa = m.penalty_score_a if m.team_a == champ_code else m.penalty_score_b
            pb = m.penalty_score_b if m.team_a == champ_code else m.penalty_score_a
            extra = f" on penalties {pa}-{pb} — the drama was UNREAL"
        elif m.extra_time:
            extra = " after a grueling extra time period"
        parts.append(
            f"And in the Grand Final — {champ_name} {score} {opp_name}{extra}! "
            f"The trophy is theirs! What an incredible tournament!"
        )

    return " ".join(parts)


def generate_mc_narration(mc_data: dict, teams: dict[str, dict]) -> str:
    """Generate a sports-commentator-style narration of Monte Carlo findings."""
    n = mc_data["n"]
    win_probs = mc_data["win_probs"]
    stage_probs = mc_data.get("stage_probs", {})
    top5 = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)[:5]

    if not top5 or top5[0][1] == 0:
        return "The simulations are in, but it's wide open — no clear favourite!"

    parts: list[str] = []
    leader_code, leader_pct = top5[0]
    leader = teams.get(leader_code, {})
    leader_name = leader.get("name", leader_code)

    # Top 5 winners with exact percentages
    parts.append(
        f"After {n:,} simulations, the numbers are IN! "
        f"Here are your top 5 title contenders:"
    )
    for i, (code, pct) in enumerate(top5):
        t = teams.get(code, {})
        name = t.get("name", code)
        flag = t.get("flag", "")
        parts.append(f"{i+1}. {flag} {name} — {pct:.1f}%")

    # Most likely final matchup
    if mc_data.get("most_likely_final"):
        pair = mc_data["most_likely_final"]
        ta = teams.get(pair[0], {})
        tb = teams.get(pair[1], {})
        parts.append(
            f"\nThe dream final? {ta.get('flag', '')} {ta.get('name', pair[0])} versus "
            f"{tb.get('name', pair[1])} {tb.get('flag', '')} — that matchup came up "
            f"{mc_data['most_likely_final_pct']:.1f}% of the time!"
        )

    # Dark horse: highest win% team ranked outside FIFA top 15
    dark_horses = [
        (code, pct) for code, pct in win_probs.items()
        if pct > 0 and teams.get(code, {}).get("fifa_ranking", 100) > 15
    ]
    if dark_horses:
        dark_horses.sort(key=lambda x: x[1], reverse=True)
        dh_code, dh_pct = dark_horses[0]
        dh = teams.get(dh_code, {})
        parts.append(
            f"\nDark horse alert! {dh.get('flag', '')} {dh.get('name', dh_code)} "
            f"(FIFA #{dh.get('fifa_ranking', '?')}) are winning it in {dh_pct:.1f}% "
            f"of simulations — don't sleep on them!"
        )

    # Semi-final regulars: top 3 teams by SF+Final+Winner combined probability
    if stage_probs:
        sf_regulars = []
        for code, probs in stage_probs.items():
            deep_run_pct = probs.get("SF", 0) + probs.get("Final", 0) + probs.get("Winner", 0)
            if deep_run_pct > 0:
                sf_regulars.append((code, deep_run_pct))
        sf_regulars.sort(key=lambda x: x[1], reverse=True)
        top3_sf = sf_regulars[:3]
        if top3_sf:
            sf_names = ", ".join(
                f"{teams.get(c, {}).get('flag', '')} {teams.get(c, {}).get('name', c)} ({p:.1f}%)"
                for c, p in top3_sf
            )
            parts.append(
                f"\nSemi-final regulars (SF+ reach rate): {sf_names}"
            )

    return "\n".join(parts)

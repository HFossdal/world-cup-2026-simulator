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

from data import TEAMS, GROUPS, get_h2h
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
div[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    background-color: #132613 !important;
}
</style>
"""


def inject_css():
    """Inject custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
):
    """Append SVG elements for one match node at (x, y)."""
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
    is_a_winner = m.winner == m.team_a
    is_b_winner = m.winner == m.team_b
    col_a = COL_WINNER if is_a_winner else COL_TEXT
    col_b = COL_WINNER if is_b_winner else COL_TEXT
    weight_a = "bold" if is_a_winner else "normal"
    weight_b = "bold" if is_b_winner else "normal"

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


def generate_bracket_svg(result: TournamentResult, teams: dict) -> str:
    """Generate a full SVG tournament bracket."""

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
            _draw_match_node(svg, m, x, y, teams)

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


def render_bracket(result: TournamentResult, teams: dict[str, dict]):
    """Render the full knockout bracket as an interactive SVG."""
    svg_html = generate_bracket_svg(result, teams)

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


def generate_narration(result: TournamentResult, teams: dict[str, dict]) -> str:
    """Generate a plain-English narration of the tournament outcome,
    tracing the champion's path from group stage to final."""
    if not result.champion:
        return "The tournament simulation completed but no champion was determined."

    champ_code = result.champion
    champ = teams.get(champ_code, {})
    parts: list[str] = []

    parts.append(
        f"{champ.get('flag', '')} {champ.get('name', champ_code)} "
        f"won the 2026 FIFA World Cup!"
    )

    # Group stage
    for group, table in result.group_tables.items():
        for i, standing in enumerate(table):
            if standing.team == champ_code:
                parts.append(
                    f"They finished {_ordinal(i + 1)} in Group {group} "
                    f"with {standing.points} points "
                    f"({standing.wins}W {standing.draws}D {standing.losses}L, "
                    f"{standing.gf} goals scored)."
                )
                break

    # Knockout path
    round_labels = [
        ("Round of 32", result.r32_matches),
        ("Round of 16", result.r16_matches),
        ("Quarterfinals", result.qf_matches),
        ("Semifinals", result.sf_matches),
    ]

    for round_name, matches in round_labels:
        for m in matches:
            if m and m.winner == champ_code:
                opp_code = m.team_b if m.team_a == champ_code else m.team_a
                opp = teams.get(opp_code, {})
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
                    extra = f" on penalties ({pa}-{pb})"
                elif m.extra_time:
                    extra = " after extra time"
                parts.append(
                    f"In the {round_name}, they beat "
                    f"{opp.get('flag', '')} {opp.get('name', opp_code)} "
                    f"{score}{extra}."
                )
                break

    # Final
    if result.final_match:
        m = result.final_match
        opp_code = m.team_b if m.team_a == champ_code else m.team_a
        opp = teams.get(opp_code, {})
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
            extra = f" on penalties ({pa}-{pb})"
        elif m.extra_time:
            extra = " after extra time"
        parts.append(
            f"In the Final, they defeated "
            f"{opp.get('flag', '')} {opp.get('name', opp_code)} "
            f"{score}{extra} to lift the trophy!"
        )

    return " ".join(parts)

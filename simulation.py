"""
World Cup 2026 Simulator — Simulation Engine

Poisson-based match simulation, group stage, knockout stage,
Monte Carlo analysis, and live-style commentary generation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from data import (
    GROUPS,
    ROUND_OF_32,
    ROUND_OF_16_FEEDS,
    QF_FEEDS,
    SF_FEEDS,
    THIRD_PLACE_SLOTS,
    assign_third_place_teams,
    get_h2h,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AVG_GOALS_PER_TEAM = 1.35  # World Cup historical average


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GoalEvent:
    minute: int
    team: str
    scorer: str
    assist: str | None = None
    is_penalty: bool = False
    is_own_goal: bool = False


@dataclass
class MatchResult:
    team_a: str
    team_b: str
    score_a: int
    score_b: int
    goals: list[GoalEvent] = field(default_factory=list)
    # Extra time / penalties
    extra_time: bool = False
    et_score_a: int = 0
    et_score_b: int = 0
    penalty_score_a: int = 0
    penalty_score_b: int = 0
    penalties: bool = False
    winner: str | None = None
    # Advanced stats (simulated)
    xg_a: float = 0.0
    xg_b: float = 0.0
    possession_a: float = 50.0
    shots_a: int = 0
    shots_b: int = 0
    shots_on_target_a: int = 0
    shots_on_target_b: int = 0
    commentary: list[str] = field(default_factory=list)


@dataclass
class GroupStanding:
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    gf: int = 0
    ga: int = 0
    points: int = 0
    results: list[MatchResult] = field(default_factory=list)

    @property
    def gd(self) -> int:
        return self.gf - self.ga


@dataclass
class TournamentResult:
    group_tables: dict[str, list[GroupStanding]]
    group_matches: dict[str, list[MatchResult]]
    third_place_ranking: list[GroupStanding]
    r32_matches: list[MatchResult]
    r16_matches: list[MatchResult]
    qf_matches: list[MatchResult]
    sf_matches: list[MatchResult]
    third_place_match: MatchResult | None
    final_match: MatchResult | None
    champion: str | None = None
    runner_up: str | None = None
    third_place: str | None = None


# ---------------------------------------------------------------------------
# Poisson match engine
# ---------------------------------------------------------------------------

def _poisson_pmf(k: int, lam: float) -> float:
    """Probability mass function of Poisson distribution."""
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def _sample_poisson(lam: float) -> int:
    """Sample from Poisson distribution."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p < L:
            return k - 1


def calculate_expected_goals(
    team_a: dict, team_b: dict, neutral: bool = True
) -> tuple[float, float]:
    """Calculate expected goals for each team using the Poisson model."""
    att_a = team_a["attack"]
    def_a = team_a["defense"]
    att_b = team_b["attack"]
    def_b = team_b["defense"]
    form_a = team_a.get("form", 0.5)
    form_b = team_b.get("form", 0.5)

    # Form adjustment (±15%)
    form_factor_a = 0.85 + 0.30 * form_a
    form_factor_b = 0.85 + 0.30 * form_b

    # Expected goals
    lambda_a = AVG_GOALS_PER_TEAM * (att_a / 1.40) * (1.40 / def_b) * form_factor_a
    lambda_b = AVG_GOALS_PER_TEAM * (att_b / 1.40) * (1.40 / def_a) * form_factor_b

    # Clamp to reasonable range
    lambda_a = max(0.3, min(4.0, lambda_a))
    lambda_b = max(0.3, min(4.0, lambda_b))

    return lambda_a, lambda_b


def _pick_scorer(team: dict, exclude: list[str] | None = None) -> str:
    """Pick a random scorer from the team's key players, weighted by position."""
    players = team.get("key_players", {})
    exclude = exclude or []
    weighted: list[tuple[str, float]] = []
    weights_map = {"FW": 3.0, "MF": 1.5, "DF": 0.4, "GK": 0.05}
    for pos, names in players.items():
        w = weights_map.get(pos, 1.0)
        for name in names:
            if name not in exclude:
                weighted.append((name, w))
    if not weighted:
        return "Unknown"
    names_list, weights = zip(*weighted)
    return random.choices(names_list, weights=weights, k=1)[0]


def _pick_assist(team: dict, scorer: str) -> str | None:
    """Pick an assist provider (different from scorer), ~60% chance of assist."""
    if random.random() > 0.60:
        return None
    players = team.get("key_players", {})
    candidates: list[str] = []
    for names in players.values():
        candidates.extend(n for n in names if n != scorer)
    return random.choice(candidates) if candidates else None


def _generate_goal_minutes(n_goals: int, start: int = 1, end: int = 90) -> list[int]:
    """Generate sorted random minutes for goals."""
    minutes = sorted(random.randint(start, end) for _ in range(n_goals))
    return minutes


def simulate_match(
    team_a_data: dict,
    team_b_data: dict,
    neutral: bool = True,
    allow_draw: bool = True,
    generate_commentary: bool = False,
) -> MatchResult:
    """Simulate a single match between two teams."""
    code_a = team_a_data["code"]
    code_b = team_b_data["code"]
    name_a = team_a_data["name"]
    name_b = team_b_data["name"]
    flag_a = team_a_data["flag"]
    flag_b = team_b_data["flag"]

    lambda_a, lambda_b = calculate_expected_goals(team_a_data, team_b_data, neutral)

    # Sample goals from Poisson
    goals_a = _sample_poisson(lambda_a)
    goals_b = _sample_poisson(lambda_b)

    # Generate goal events
    all_goals: list[GoalEvent] = []
    minutes_a = _generate_goal_minutes(goals_a)
    minutes_b = _generate_goal_minutes(goals_b)

    for minute in minutes_a:
        scorer = _pick_scorer(team_a_data)
        assist = _pick_assist(team_a_data, scorer)
        all_goals.append(GoalEvent(
            minute=minute, team=code_a, scorer=scorer, assist=assist
        ))

    for minute in minutes_b:
        scorer = _pick_scorer(team_b_data)
        assist = _pick_assist(team_b_data, scorer)
        all_goals.append(GoalEvent(
            minute=minute, team=code_b, scorer=scorer, assist=assist
        ))

    all_goals.sort(key=lambda g: g.minute)

    # Simulate stats
    mid_a = team_a_data.get("midfield", 1.0)
    mid_b = team_b_data.get("midfield", 1.0)
    possession_a = round(50 * mid_a / (mid_a + mid_b) * 2, 1)
    possession_a = max(25.0, min(75.0, possession_a))

    shots_a = max(goals_a, int(lambda_a * random.uniform(3.5, 5.5)))
    shots_b = max(goals_b, int(lambda_b * random.uniform(3.5, 5.5)))
    sot_a = max(goals_a, int(shots_a * random.uniform(0.3, 0.5)))
    sot_b = max(goals_b, int(shots_b * random.uniform(0.3, 0.5)))

    result = MatchResult(
        team_a=code_a,
        team_b=code_b,
        score_a=goals_a,
        score_b=goals_b,
        goals=all_goals,
        xg_a=round(lambda_a, 2),
        xg_b=round(lambda_b, 2),
        possession_a=possession_a,
        shots_a=shots_a,
        shots_b=shots_b,
        shots_on_target_a=sot_a,
        shots_on_target_b=sot_b,
    )

    # Handle knockout draws
    if not allow_draw and goals_a == goals_b:
        # Extra time
        et_lambda_a = lambda_a * 0.33  # 30 min period
        et_lambda_b = lambda_b * 0.33
        et_goals_a = _sample_poisson(et_lambda_a)
        et_goals_b = _sample_poisson(et_lambda_b)

        et_minutes_a = _generate_goal_minutes(et_goals_a, 91, 120)
        et_minutes_b = _generate_goal_minutes(et_goals_b, 91, 120)

        for minute in et_minutes_a:
            scorer = _pick_scorer(team_a_data)
            assist = _pick_assist(team_a_data, scorer)
            all_goals.append(GoalEvent(
                minute=minute, team=code_a, scorer=scorer, assist=assist
            ))
        for minute in et_minutes_b:
            scorer = _pick_scorer(team_b_data)
            assist = _pick_assist(team_b_data, scorer)
            all_goals.append(GoalEvent(
                minute=minute, team=code_b, scorer=scorer, assist=assist
            ))
        all_goals.sort(key=lambda g: g.minute)

        result.extra_time = True
        result.et_score_a = et_goals_a
        result.et_score_b = et_goals_b
        result.goals = all_goals

        total_a = goals_a + et_goals_a
        total_b = goals_b + et_goals_b

        if total_a == total_b:
            # Penalty shootout
            result.penalties = True
            pen_a, pen_b = _simulate_penalties(team_a_data, team_b_data)
            result.penalty_score_a = pen_a
            result.penalty_score_b = pen_b
            result.winner = code_a if pen_a > pen_b else code_b
        else:
            result.winner = code_a if total_a > total_b else code_b
    else:
        if goals_a > goals_b:
            result.winner = code_a
        elif goals_b > goals_a:
            result.winner = code_b
        else:
            result.winner = None  # Draw

    # Generate commentary
    if generate_commentary:
        result.commentary = _generate_commentary(result, team_a_data, team_b_data)

    return result


def _simulate_penalties(team_a: dict, team_b: dict) -> tuple[int, int]:
    """Simulate a penalty shootout. Returns (score_a, score_b)."""
    # Base conversion rate ~75%, adjusted by team quality
    rate_a = 0.70 + 0.05 * min(team_a.get("attack", 1.0), 2.0)
    rate_b = 0.70 + 0.05 * min(team_b.get("attack", 1.0), 2.0)

    score_a = 0
    score_b = 0
    # Best of 5 rounds
    for i in range(5):
        if random.random() < rate_a:
            score_a += 1
        if random.random() < rate_b:
            score_b += 1
        # Check if shootout is decided early
        remaining = 4 - i
        if score_a > score_b + remaining:
            break
        if score_b > score_a + remaining:
            break

    # Sudden death if tied after 5
    while score_a == score_b:
        if random.random() < rate_a:
            score_a += 1
        if random.random() < rate_b:
            score_b += 1
        if score_a != score_b:
            break
        # Both scored or both missed → continue

    return score_a, score_b


def _generate_commentary(
    result: MatchResult, team_a: dict, team_b: dict
) -> list[str]:
    """Generate live-style text commentary for a match."""
    lines: list[str] = []
    name_a = team_a["name"]
    name_b = team_b["name"]
    flag_a = team_a["flag"]
    flag_b = team_b["flag"]

    lines.append(f"**{flag_a} {name_a} vs {name_b} {flag_b}**")
    lines.append("---")

    score_a, score_b = 0, 0
    for goal in result.goals:
        if goal.team == result.team_a:
            score_a += 1
            team_name = name_a
            flag = flag_a
        else:
            score_b += 1
            team_name = name_b
            flag = flag_b

        minute_str = f"{goal.minute}'"
        if goal.minute > 90:
            minute_str = f"{goal.minute}' (ET)"

        assist_str = f" (assist: {goal.assist})" if goal.assist else ""
        lines.append(
            f"⚽ **{minute_str}** — {flag} **{goal.scorer}** scores for "
            f"{team_name}!{assist_str}  [{score_a}-{score_b}]"
        )

    if not result.goals:
        lines.append("A tightly contested match with no goals.")

    lines.append("---")

    # Final score line
    total_a = result.score_a + result.et_score_a
    total_b = result.score_b + result.et_score_b
    score_line = f"**Full Time: {flag_a} {name_a} {total_a} - {total_b} {name_b} {flag_b}**"

    if result.extra_time:
        score_line += f"  (after extra time: {result.score_a}-{result.score_b} at 90')"
    if result.penalties:
        score_line += f"\n**Penalties: {result.penalty_score_a} - {result.penalty_score_b}**"

    lines.append(score_line)
    return lines


# ---------------------------------------------------------------------------
# Group stage simulation
# ---------------------------------------------------------------------------

def simulate_group_stage(
    teams: dict[str, dict],
    groups: dict[str, list[str]] | None = None,
    locked_results: dict[tuple[str, str], tuple[int, int]] | None = None,
) -> tuple[dict[str, list[GroupStanding]], dict[str, list[MatchResult]]]:
    """Simulate the full group stage. Returns tables and match results."""
    groups = groups or GROUPS
    locked_results = locked_results or {}
    all_tables: dict[str, list[GroupStanding]] = {}
    all_matches: dict[str, list[MatchResult]] = {}

    for group_letter, team_codes in groups.items():
        standings = {code: GroupStanding(team=code) for code in team_codes}
        matches: list[MatchResult] = []

        # Round-robin: each team plays every other team
        for i in range(len(team_codes)):
            for j in range(i + 1, len(team_codes)):
                code_a = team_codes[i]
                code_b = team_codes[j]
                team_a = teams[code_a]
                team_b = teams[code_b]

                # Check for locked results
                lock_key = (code_a, code_b)
                rev_key = (code_b, code_a)

                if lock_key in locked_results:
                    sa, sb = locked_results[lock_key]
                    result = MatchResult(
                        team_a=code_a, team_b=code_b,
                        score_a=sa, score_b=sb,
                        winner=code_a if sa > sb else (code_b if sb > sa else None),
                    )
                elif rev_key in locked_results:
                    sb, sa = locked_results[rev_key]
                    result = MatchResult(
                        team_a=code_a, team_b=code_b,
                        score_a=sa, score_b=sb,
                        winner=code_a if sa > sb else (code_b if sb > sa else None),
                    )
                else:
                    result = simulate_match(team_a, team_b, neutral=True, allow_draw=True)

                matches.append(result)

                # Update standings
                sa_standing = standings[code_a]
                sb_standing = standings[code_b]
                sa_standing.played += 1
                sb_standing.played += 1
                sa_standing.gf += result.score_a
                sa_standing.ga += result.score_b
                sb_standing.gf += result.score_b
                sb_standing.ga += result.score_a
                sa_standing.results.append(result)
                sb_standing.results.append(result)

                if result.score_a > result.score_b:
                    sa_standing.wins += 1
                    sa_standing.points += 3
                    sb_standing.losses += 1
                elif result.score_b > result.score_a:
                    sb_standing.wins += 1
                    sb_standing.points += 3
                    sa_standing.losses += 1
                else:
                    sa_standing.draws += 1
                    sb_standing.draws += 1
                    sa_standing.points += 1
                    sb_standing.points += 1

        # Sort standings: points, GD, GF, FIFA ranking (tiebreaker)
        table = sorted(
            standings.values(),
            key=lambda s: (
                s.points,
                s.gd,
                s.gf,
                -teams[s.team].get("fifa_ranking", 100),
            ),
            reverse=True,
        )
        all_tables[group_letter] = table
        all_matches[group_letter] = matches

    return all_tables, all_matches


def get_best_third_place(
    tables: dict[str, list[GroupStanding]],
    teams: dict[str, dict],
) -> list[GroupStanding]:
    """Get the 8 best third-placed teams from 12 groups."""
    third_placed = []
    for group_letter, table in tables.items():
        if len(table) >= 3:
            standing = table[2]  # 0-indexed: 3rd place
            third_placed.append((group_letter, standing))

    # Sort by: points, GD, GF, FIFA ranking
    third_placed.sort(
        key=lambda x: (
            x[1].points,
            x[1].gd,
            x[1].gf,
            -teams[x[1].team].get("fifa_ranking", 100),
        ),
        reverse=True,
    )
    return [s for _, s in third_placed[:8]]


# ---------------------------------------------------------------------------
# Knockout stage simulation
# ---------------------------------------------------------------------------

def _resolve_slot(
    slot: str,
    tables: dict[str, list[GroupStanding]],
    third_place_assignment: dict[int, str],
    match_id: int | None = None,
) -> str:
    """Resolve a bracket slot like '1A', '2B', or '3_ABCDF' to a team code."""
    if slot.startswith("3_"):
        # Third-place slot — use assignment
        if match_id and match_id in third_place_assignment:
            group = third_place_assignment[match_id]
            return tables[group][2].team  # 3rd-place team from that group
        return "TBD"
    position = int(slot[0]) - 1  # '1' → index 0, '2' → index 1
    group = slot[1]
    return tables[group][position].team


def simulate_knockout_stage(
    teams: dict[str, dict],
    tables: dict[str, list[GroupStanding]],
    third_place_standings: list[GroupStanding],
) -> TournamentResult:
    """Simulate the entire knockout stage from R32 to Final."""

    # Determine which groups' third-place teams qualified
    third_groups: list[str] = []
    for group_letter, table in tables.items():
        if len(table) >= 3:
            third_team = table[2].team
            if third_team in [s.team for s in third_place_standings]:
                third_groups.append(group_letter)

    third_place_assignment = assign_third_place_teams(sorted(third_groups))

    # --- Round of 32 ---
    r32_matches: list[MatchResult] = []
    r32_winners: dict[int, str] = {}

    for match in ROUND_OF_32:
        team_a_code = _resolve_slot(match["slot_a"], tables, third_place_assignment, match["id"])
        team_b_code = _resolve_slot(match["slot_b"], tables, third_place_assignment, match["id"])

        if team_a_code == "TBD" or team_b_code == "TBD":
            continue

        result = simulate_match(
            teams[team_a_code], teams[team_b_code],
            neutral=True, allow_draw=False,
        )
        r32_matches.append(result)
        r32_winners[match["id"]] = result.winner

    # --- Round of 16 ---
    r32_match_ids = [m["id"] for m in ROUND_OF_32]
    r16_matches: list[MatchResult] = []
    r16_winners: list[str] = []

    for idx_a, idx_b in ROUND_OF_16_FEEDS:
        mid_a = r32_match_ids[idx_a] if idx_a < len(r32_match_ids) else None
        mid_b = r32_match_ids[idx_b] if idx_b < len(r32_match_ids) else None

        winner_a = r32_winners.get(mid_a)
        winner_b = r32_winners.get(mid_b)

        if not winner_a or not winner_b:
            continue

        result = simulate_match(
            teams[winner_a], teams[winner_b],
            neutral=True, allow_draw=False,
        )
        r16_matches.append(result)
        r16_winners.append(result.winner)

    # --- Quarterfinals ---
    qf_matches: list[MatchResult] = []
    qf_winners: list[str] = []

    for idx_a, idx_b in QF_FEEDS:
        if idx_a < len(r16_winners) and idx_b < len(r16_winners):
            result = simulate_match(
                teams[r16_winners[idx_a]], teams[r16_winners[idx_b]],
                neutral=True, allow_draw=False,
            )
            qf_matches.append(result)
            qf_winners.append(result.winner)

    # --- Semifinals ---
    sf_matches: list[MatchResult] = []
    sf_winners: list[str] = []
    sf_losers: list[str] = []

    for idx_a, idx_b in SF_FEEDS:
        if idx_a < len(qf_winners) and idx_b < len(qf_winners):
            result = simulate_match(
                teams[qf_winners[idx_a]], teams[qf_winners[idx_b]],
                neutral=True, allow_draw=False,
            )
            sf_matches.append(result)
            sf_winners.append(result.winner)
            loser = result.team_a if result.winner == result.team_b else result.team_b
            sf_losers.append(loser)

    # --- Third-place match ---
    third_place_match = None
    if len(sf_losers) == 2:
        third_place_match = simulate_match(
            teams[sf_losers[0]], teams[sf_losers[1]],
            neutral=True, allow_draw=False,
        )

    # --- Final ---
    final_match = None
    champion = None
    runner_up = None

    if len(sf_winners) == 2:
        final_match = simulate_match(
            teams[sf_winners[0]], teams[sf_winners[1]],
            neutral=True, allow_draw=False,
            generate_commentary=True,
        )
        champion = final_match.winner
        runner_up = final_match.team_a if champion == final_match.team_b else final_match.team_b

    third_place_team = third_place_match.winner if third_place_match else None

    return TournamentResult(
        group_tables=tables,
        group_matches={},  # filled by caller
        third_place_ranking=third_place_standings,
        r32_matches=r32_matches,
        r16_matches=r16_matches,
        qf_matches=qf_matches,
        sf_matches=sf_matches,
        third_place_match=third_place_match,
        final_match=final_match,
        champion=champion,
        runner_up=runner_up,
        third_place=third_place_team,
    )


# ---------------------------------------------------------------------------
# Full tournament simulation
# ---------------------------------------------------------------------------

def simulate_tournament(
    teams: dict[str, dict],
    groups: dict[str, list[str]] | None = None,
    locked_results: dict | None = None,
) -> TournamentResult:
    """Simulate the full FIFA World Cup 2026 tournament."""
    tables, group_matches = simulate_group_stage(teams, groups, locked_results)
    third_place_standings = get_best_third_place(tables, teams)

    result = simulate_knockout_stage(teams, tables, third_place_standings)
    result.group_matches = group_matches
    result.group_tables = tables
    result.third_place_ranking = third_place_standings

    return result


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def run_monte_carlo(
    teams: dict[str, dict],
    n: int = 1000,
    groups: dict[str, list[str]] | None = None,
    locked_results: dict | None = None,
) -> dict:
    """Run n simulations and return probability data for each team."""
    groups = groups or GROUPS

    # Track how far each team goes
    stages = ["Group Exit", "R32", "R16", "QF", "SF", "Final", "Winner"]
    all_team_codes = list(teams.keys())
    counts: dict[str, dict[str, int]] = {
        code: {s: 0 for s in stages} for code in all_team_codes
    }
    win_counts: dict[str, int] = {code: 0 for code in all_team_codes}
    final_matchups: dict[tuple[str, str], int] = {}

    for _ in range(n):
        result = simulate_tournament(teams, groups, locked_results)

        # Determine which teams reached each stage
        r32_teams = set()
        for m in result.r32_matches:
            r32_teams.add(m.team_a)
            r32_teams.add(m.team_b)

        r16_teams = set()
        for m in result.r16_matches:
            r16_teams.add(m.team_a)
            r16_teams.add(m.team_b)

        qf_teams = set()
        for m in result.qf_matches:
            qf_teams.add(m.team_a)
            qf_teams.add(m.team_b)

        sf_teams = set()
        for m in result.sf_matches:
            sf_teams.add(m.team_a)
            sf_teams.add(m.team_b)

        finalist_teams = set()
        if result.final_match:
            finalist_teams.add(result.final_match.team_a)
            finalist_teams.add(result.final_match.team_b)
            # Track final matchup
            pair = tuple(sorted([result.final_match.team_a, result.final_match.team_b]))
            final_matchups[pair] = final_matchups.get(pair, 0) + 1

        for code in all_team_codes:
            if result.champion == code:
                counts[code]["Winner"] += 1
            elif code in finalist_teams:
                counts[code]["Final"] += 1
            elif code in sf_teams:
                counts[code]["SF"] += 1
            elif code in qf_teams:
                counts[code]["QF"] += 1
            elif code in r16_teams:
                counts[code]["R16"] += 1
            elif code in r32_teams:
                counts[code]["R32"] += 1
            else:
                counts[code]["Group Exit"] += 1

        if result.champion:
            win_counts[result.champion] = win_counts.get(result.champion, 0) + 1

    # Convert to probabilities
    probs: dict[str, dict[str, float]] = {}
    for code in all_team_codes:
        probs[code] = {s: round(counts[code][s] / n * 100, 1) for s in stages}

    win_probs = {
        code: round(win_counts[code] / n * 100, 1) for code in all_team_codes
    }

    # Most likely final
    most_likely_final = max(final_matchups, key=final_matchups.get) if final_matchups else None
    most_likely_final_pct = (
        round(final_matchups[most_likely_final] / n * 100, 1)
        if most_likely_final
        else 0
    )

    return {
        "n": n,
        "stage_probs": probs,
        "win_probs": win_probs,
        "most_likely_final": most_likely_final,
        "most_likely_final_pct": most_likely_final_pct,
        "final_matchups": final_matchups,
    }

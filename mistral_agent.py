"""
World Cup 2026 Simulator — Mistral AI Scenario Agent

Natural-language chat interface powered by Mistral that lets users
modify simulation parameters through conversation.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from data import TEAMS, GROUPS, get_teams_copy, CONFIRMED_QUALIFIED, PLAYOFF_SLOTS

# ---------------------------------------------------------------------------
# Try to import Mistral client
# ---------------------------------------------------------------------------
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# System prompt for Mistral
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """You are the World Cup 2026 Simulator AI, an expert FIFA World Cup 2026 analyst assistant.
You help users explore "what if" scenarios for the World Cup by modifying simulation parameters.

When users ask about scenarios, you MUST respond with a JSON block wrapped in ```json ... ``` tags
containing the modifications, followed by a plain-English explanation.

## Available modification types:

1. **adjust_team_rating** — Change a team's attack, defense, or midfield rating.
   ```json
   {{"action": "adjust_team_rating", "team": "FRA", "attribute": "attack", "delta": -0.20}}
   ```

2. **lock_result** — Lock a specific match result.
   ```json
   {{"action": "lock_result", "team_a": "NOR", "team_b": "FRA", "score_a": 2, "score_b": 1}}
   ```

3. **boost_team** — Increase all ratings for a team by a percentage.
   ```json
   {{"action": "boost_team", "team": "BRA", "pct": 15}}
   ```

4. **nerf_team** — Decrease all ratings for a team by a percentage.
   ```json
   {{"action": "nerf_team", "team": "BRA", "pct": 20}}
   ```

5. **simulate** — Trigger a simulation (single or Monte Carlo).
   ```json
   {{"action": "simulate", "mode": "once"}}
   ```
   or:
   ```json
   {{"action": "simulate", "mode": "monte_carlo", "n": 1000}}
   ```

6. **reset** — Reset all modifications back to baseline.
   ```json
   {{"action": "reset"}}
   ```

You can chain multiple actions in an array: ```json [action1, action2, ...]```.

## Team codes (use these exactly):
{team_codes}

## Group draw:
{group_draw}

## Guidelines:
- For injury scenarios, reduce the team's attack (for forwards) or defense (for defenders) by 0.10-0.25.
- For "what if they win", lock the result and suggest re-simulating.
- For probability questions, suggest running Monte Carlo (1000 sims).
- Always explain your reasoning clearly.
- **Write like a sports commentator!** Be excited, dynamic, and enthusiastic. Use punchy language and football terminology. Never be robotic or dry. Your responses will be spoken aloud by a voice narrator, so make them sound natural and thrilling.
- Example tone: "Great shout! Let's see what happens if the Seleção get a 20% power boost — they'll be absolutely terrifying going forward!"
- If user wants to reset, include the reset action.
"""


def _build_system_prompt(
    team_codes: list[str] | None = None,
    groups: dict[str, list[str]] | None = None,
) -> str:
    """Build a system prompt dynamically from the active teams and groups."""
    if team_codes is None:
        team_codes = sorted(TEAMS.keys())
    if groups is None:
        groups = GROUPS

    codes_str = ", ".join(sorted(team_codes))

    group_parts: list[str] = []
    letters = sorted(groups.keys())
    for i in range(0, len(letters), 4):
        chunk = letters[i:i+4]
        group_parts.append("  |  ".join(
            f"{l}: {', '.join(groups[l])}" for l in chunk
        ))
    draw_str = "\n".join(group_parts)

    return _SYSTEM_PROMPT_TEMPLATE.format(
        team_codes=codes_str,
        group_draw=draw_str,
    )


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

@dataclass
class ScenarioModification:
    action: str
    params: dict = field(default_factory=dict)


@dataclass
class AgentResponse:
    message: str
    modifications: list[ScenarioModification] = field(default_factory=list)
    should_simulate: bool = False
    sim_mode: str = "once"
    sim_n: int = 1000
    should_reset: bool = False


class MistralScenarioAgent:
    """Mistral-powered scenario agent for modifying World Cup simulations."""

    def __init__(
        self,
        team_codes: list[str] | None = None,
        groups: dict[str, list[str]] | None = None,
    ):
        self.api_key = os.getenv("MISTRAL_API_KEY", "")
        self.model = "mistral-large-latest"
        self.conversation_history: list[dict] = []
        self.modifications_log: list[str] = []
        self.client = None
        self.system_prompt = _build_system_prompt(team_codes, groups)

        if MISTRAL_AVAILABLE and self.api_key:
            self.client = Mistral(api_key=self.api_key)

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def reset_conversation(self):
        self.conversation_history = []
        self.modifications_log = []

    def chat(self, user_message: str, current_teams: dict[str, dict]) -> AgentResponse:
        """Send a message to Mistral and parse the response for modifications."""
        if not self.is_available:
            return self._fallback_parse(user_message, current_teams)

        self.conversation_history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history[-20:])  # Keep last 20 messages

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1500,
            )
            assistant_msg = response.choices[0].message.content
        except Exception as e:
            assistant_msg = f"I encountered an error connecting to Mistral: {str(e)}. Let me try to help with a basic interpretation."
            return self._fallback_parse(user_message, current_teams, error_msg=assistant_msg)

        self.conversation_history.append({"role": "assistant", "content": assistant_msg})

        return self._parse_response(assistant_msg)

    def _parse_response(self, response_text: str) -> AgentResponse:
        """Parse Mistral's response for JSON modification blocks."""
        modifications: list[ScenarioModification] = []
        should_simulate = False
        sim_mode = "once"
        sim_n = 1000
        should_reset = False

        # Extract JSON blocks
        json_pattern = r"```json\s*([\s\S]*?)\s*```"
        json_matches = re.findall(json_pattern, response_text)

        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if isinstance(data, list):
                    actions = data
                else:
                    actions = [data]

                for action_data in actions:
                    action = action_data.get("action", "")

                    if action == "reset":
                        should_reset = True
                    elif action == "simulate":
                        should_simulate = True
                        sim_mode = action_data.get("mode", "once")
                        sim_n = action_data.get("n", 1000)
                    else:
                        modifications.append(ScenarioModification(
                            action=action,
                            params={k: v for k, v in action_data.items() if k != "action"},
                        ))
            except json.JSONDecodeError:
                continue

        # Remove JSON blocks from display message
        display_text = re.sub(json_pattern, "", response_text).strip()

        return AgentResponse(
            message=display_text,
            modifications=modifications,
            should_simulate=should_simulate,
            sim_mode=sim_mode,
            sim_n=sim_n,
            should_reset=should_reset,
        )

    def _fallback_parse(
        self, user_message: str, current_teams: dict, error_msg: str = ""
    ) -> AgentResponse:
        """Basic keyword-based fallback when Mistral is unavailable."""
        msg_lower = user_message.lower()
        modifications: list[ScenarioModification] = []
        should_simulate = False
        sim_mode = "once"
        sim_n = 1000
        should_reset = False
        explanation_parts: list[str] = []

        if "reset" in msg_lower:
            should_reset = True
            explanation_parts.append("Resetting all modifications back to baseline.")

        # Detect injury scenarios
        injury_match = re.search(
            r"([\w\s]+)\s+(?:is injured|injured|misses|out|absent)", msg_lower
        )
        if injury_match:
            # Try to find the team
            for code, team in current_teams.items():
                for pos_players in team.get("key_players", {}).values():
                    for player in pos_players:
                        if player.lower() in msg_lower:
                            attr = "attack" if any(
                                player in team.get("key_players", {}).get("FW", [])
                            ) else "defense"
                            modifications.append(ScenarioModification(
                                action="adjust_team_rating",
                                params={"team": code, "attribute": attr, "delta": -0.15},
                            ))
                            explanation_parts.append(
                                f"Reducing {team['name']}'s {attr} by 0.15 to simulate "
                                f"the impact of losing {player}."
                            )
                            break

        # Detect strength boost/nerf
        stronger_match = re.search(
            r"make\s+([\w\s]+)\s+(\d+)%?\s+stronger", msg_lower
        )
        if stronger_match:
            team_name = stronger_match.group(1).strip()
            pct = int(stronger_match.group(2))
            for code, team in current_teams.items():
                if team["name"].lower() == team_name.lower():
                    modifications.append(ScenarioModification(
                        action="boost_team",
                        params={"team": code, "pct": pct},
                    ))
                    explanation_parts.append(
                        f"Boosting {team['name']}'s ratings by {pct}%."
                    )
                    break

        weaker_match = re.search(
            r"make\s+([\w\s]+)\s+(\d+)%?\s+weaker", msg_lower
        )
        if weaker_match:
            team_name = weaker_match.group(1).strip()
            pct = int(weaker_match.group(2))
            for code, team in current_teams.items():
                if team["name"].lower() == team_name.lower():
                    modifications.append(ScenarioModification(
                        action="nerf_team",
                        params={"team": code, "pct": pct},
                    ))
                    explanation_parts.append(
                        f"Reducing {team['name']}'s ratings by {pct}%."
                    )
                    break

        # Detect simulation requests
        if any(w in msg_lower for w in ["simulate", "run", "chances", "probability", "likely", "winners", "predict"]):
            should_simulate = True
            if any(w in msg_lower for w in ["1000", "monte carlo", "chances", "probability", "likely"]):
                sim_mode = "monte_carlo"
                explanation_parts.append("Running 1000 Monte Carlo simulations...")
            else:
                sim_mode = "once"
                explanation_parts.append("Running a single tournament simulation...")

        # Detect result locking
        win_match = re.search(
            r"(?:what if|if)\s+([\w\s]+)\s+(?:beats?|wins?|defeats?)\s+([\w\s]+)", msg_lower
        )
        if win_match:
            team_a_name = win_match.group(1).strip()
            team_b_name = win_match.group(2).strip()
            for code_a, data_a in current_teams.items():
                if data_a["name"].lower() == team_a_name.lower():
                    for code_b, data_b in current_teams.items():
                        if data_b["name"].lower() == team_b_name.lower():
                            modifications.append(ScenarioModification(
                                action="lock_result",
                                params={
                                    "team_a": code_a,
                                    "team_b": code_b,
                                    "score_a": 2,
                                    "score_b": 1,
                                },
                            ))
                            explanation_parts.append(
                                f"Locking result: {data_a['name']} 2-1 {data_b['name']}."
                            )
                            should_simulate = True
                            sim_mode = "monte_carlo"
                            break
                    break

        if not explanation_parts:
            if error_msg:
                explanation_parts.append(error_msg)
            else:
                explanation_parts.append(
                    "I couldn't parse a specific scenario from your message. "
                    "Try something like:\n"
                    "- 'What if Norway beats France?'\n"
                    "- 'Make Brazil 20% stronger'\n"
                    "- 'What if Mbappé is injured?'\n"
                    "- 'Show me the 10 most likely winners'\n"
                    "- 'Reset to baseline'\n\n"
                    "**Note:** Set your MISTRAL_API_KEY in .env for full AI-powered scenario analysis!"
                )

        return AgentResponse(
            message="\n".join(explanation_parts),
            modifications=modifications,
            should_simulate=should_simulate,
            sim_mode=sim_mode,
            sim_n=sim_n,
            should_reset=should_reset,
        )


def apply_modifications(
    teams: dict[str, dict],
    modifications: list[ScenarioModification],
) -> tuple[dict[str, dict], list[str], dict[tuple[str, str], tuple[int, int]]]:
    """Apply scenario modifications to team data. Returns modified teams,
    a log of changes, and any locked results."""
    locked_results: dict[tuple[str, str], tuple[int, int]] = {}
    change_log: list[str] = []

    for mod in modifications:
        action = mod.action
        p = mod.params

        if action == "adjust_team_rating":
            code = p.get("team", "")
            attr = p.get("attribute", "attack")
            delta = p.get("delta", 0)
            if code in teams and attr in ("attack", "defense", "midfield"):
                old = teams[code][attr]
                teams[code][attr] = max(0.5, min(2.5, old + delta))
                change_log.append(
                    f"{teams[code]['name']}: {attr} {old:.2f} → {teams[code][attr]:.2f}"
                )

        elif action == "boost_team":
            code = p.get("team", "")
            pct = p.get("pct", 10) / 100
            if code in teams:
                for attr in ("attack", "defense", "midfield"):
                    old = teams[code][attr]
                    teams[code][attr] = min(2.5, old * (1 + pct))
                change_log.append(
                    f"{teams[code]['name']}: all ratings boosted by {p.get('pct', 10)}%"
                )

        elif action == "nerf_team":
            code = p.get("team", "")
            pct = p.get("pct", 10) / 100
            if code in teams:
                for attr in ("attack", "defense", "midfield"):
                    old = teams[code][attr]
                    teams[code][attr] = max(0.5, old * (1 - pct))
                change_log.append(
                    f"{teams[code]['name']}: all ratings reduced by {p.get('pct', 10)}%"
                )

        elif action == "lock_result":
            ta = p.get("team_a", "")
            tb = p.get("team_b", "")
            sa = p.get("score_a", 0)
            sb = p.get("score_b", 0)
            if ta and tb:
                locked_results[(ta, tb)] = (sa, sb)
                ta_name = teams.get(ta, {}).get("name", ta)
                tb_name = teams.get(tb, {}).get("name", tb)
                change_log.append(f"Locked result: {ta_name} {sa}-{sb} {tb_name}")

    return teams, change_log, locked_results

"""
llm_rrt_planner.py
===================
Coarse-to-fine path planning: an LLM proposes a short sequence of guidepost
waypoints from the current position to the target, and RRT-Connect plans the
fine-grained route between each consecutive pair. If a guidepost lies inside
an obstacle, or RRT-Connect cannot connect to it within its sample budget,
the reason is fed back to the LLM and the remaining route is re-planned --
never restarted from the beginning.
"""

import json
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from groq import Groq

from .obstacles_2d import compute_local_free_rects
from .rrt_planner import RRTConnectPlanner

Point = Tuple[float, float]

_llm_state = {"api_key": "", "model": "openai/gpt-oss-120b", "temperature": 0.4, "max_tokens": 2048}
_client: Optional[Groq] = None


def set_llm_config(api_key: str, model: str, temperature: float, max_tokens: int) -> None:
    global _client
    _llm_state.update(api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens)
    _client = Groq(api_key=api_key)


def _llm_call(system_prompt: str, user_prompt: str) -> str:
    chunks = _client.chat.completions.create(
        model=_llm_state["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=_llm_state["temperature"],
        max_completion_tokens=_llm_state["max_tokens"],
        reasoning_effort="low",  # GPT-OSS-specific: keep the token budget for the JSON answer, not reasoning
        top_p=1,
        stream=True,
        stop=None,
    )
    result = ""
    for chunk in chunks:
        result += chunk.choices[0].delta.content or ""
    return result.strip()


def parse_waypoints(raw: str) -> Optional[List[Point]]:
    text = re.sub(r"```[a-z]*", "", raw).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return [(float(p[0]), float(p[1])) for p in data.get("waypoints", [])]
    except Exception:
        return None


def clamp_waypoints(wps: List[Point], room_bounds: Tuple[float, float]) -> List[Point]:
    W, D = room_bounds
    return [(max(0.05, min(W - 0.05, x)), max(0.05, min(D - 0.05, y))) for x, y in wps]


@dataclass
class RouteResult:
    path: List[Point]
    reached: bool
    llm_calls: int
    conversation_log: List[dict] = field(default_factory=list)


def plan_full_route(system_prompt: str, planner: RRTConnectPlanner, start: Point, target: Point,
                     max_llm_retries: int = 6, initial_context: Optional[str] = None) -> RouteResult:
    obstacle_map, room_bounds = planner.obstacle_map, planner.room_bounds
    path: List[Point] = [start]
    current_pos = start
    rejection_reason = initial_context
    conversation_log: List[dict] = []
    llm_calls = 0

    while llm_calls < max_llm_retries:
        user_prompt = json.dumps({
            "start": [round(v, 2) for v in current_pos],
            "target": [round(v, 2) for v in target],
            "rejection": rejection_reason,
        }, separators=(",", ":"))

        raw = _llm_call(system_prompt, user_prompt)
        llm_calls += 1
        conversation_log.append({"llm_call": llm_calls, "user_prompt": user_prompt, "raw_response": raw})

        waypoints = parse_waypoints(raw)
        if waypoints is None:
            rejection_reason = "Response not parseable as JSON. Return ONLY the JSON object."
            continue

        waypoints = clamp_waypoints(waypoints, room_bounds)
        if not waypoints or waypoints[-1] != target:
            waypoints.append(target)

        segment_ok = True
        rejection_reason = None

        for seg_idx, sub_goal in enumerate(waypoints):
            if math.dist(current_pos, target) < planner.step_size:
                path.append(target)
                return RouteResult(path, True, llm_calls, conversation_log)

            if not obstacle_map.is_point_free(sub_goal):
                local_rects = compute_local_free_rects(obstacle_map, sub_goal, room_bounds)
                hint = f" Nearby free zones: {json.dumps(local_rects, separators=(',', ':'))}." if local_rects else ""
                n_remaining = len(waypoints) - seg_idx
                rejection_reason = (
                    f"Waypoint {seg_idx + 1} at {[round(v, 2) for v in sub_goal]} is inside an "
                    f"obstacle.{hint} Current safe position: {[round(v, 2) for v in current_pos]}. "
                    f"Generate exactly {n_remaining} new waypoints avoiding this region."
                )
                segment_ok = False
                break

            segment = planner.plan(current_pos, sub_goal)
            if segment is None:
                n_remaining = len(waypoints) - seg_idx
                rejection_reason = (
                    f"Could not find a route to waypoint {seg_idx + 1} at "
                    f"{[round(v, 2) for v in sub_goal]}. "
                    f"Current position: {[round(v, 2) for v in current_pos]}. "
                    f"Generate exactly {n_remaining} new waypoints, preferably through a wider gap."
                )
                segment_ok = False
                break

            path.extend(segment[1:])
            current_pos = sub_goal

        if segment_ok:
            if math.dist(current_pos, target) < planner.step_size:
                return RouteResult(path, True, llm_calls, conversation_log)
            rejection_reason = (
                f"Path exhausted but target not reached. Current: {[round(v, 2) for v in current_pos]}. "
                "Continue from current position."
            )

    return RouteResult(path, False, llm_calls, conversation_log)
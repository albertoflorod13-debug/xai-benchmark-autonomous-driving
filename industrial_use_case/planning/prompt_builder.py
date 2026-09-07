"""
prompt_builder.py
==================
Builds the system prompt sent to the LLM: room bounds, the exact list of
obstacle boxes to avoid, and the waypoint-proposal contract (JSON schema,
step size, waypoint count range, last waypoint = target).
"""

import json
from typing import List, Optional, Tuple

Box = Tuple[float, float, float, float]

WAYPOINT_COUNT_RANGE = (2, 5)


def build_system_prompt(room_bounds: Tuple[float, float], obstacles: List[Box],
                         step_size: float, extra_rules: Optional[str] = None) -> str:
    W, D = room_bounds
    lo, hi = WAYPOINT_COUNT_RANGE
    prompt = (
        f"Robot path planner in a {W:.0f}x{D:.0f} m warehouse room.\n"
        "Respond ONLY with JSON: {\"waypoints\":[[x,y],...]} -- no markdown.\n"
        f"Rules: stay in room bounds (x:0-{W:g}, y:0-{D:g}); "
        f"max step {step_size:g} m; last waypoint = target; {lo}-{hi} waypoints."
    )
    if extra_rules:
        prompt += f"\nAdditional guidance: {extra_rules}"

    if obstacles:
        boxes_fmt = [{"x": [b[0], b[2]], "y": [b[1], b[3]]} for b in obstacles]
        prompt += (
            "\nSolid obstacles in the room (axis-aligned boxes you must never enter "
            "or cross) -- for each box, x/y give the [min, max] extent on that axis. "
            "Study the gaps between them before proposing waypoints: "
            f"{json.dumps(boxes_fmt, separators=(',', ':'))}"
        )
    return prompt
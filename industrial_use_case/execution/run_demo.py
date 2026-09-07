"""
run_demo.py
============
Runs one full offline-plan + online-execution demo: plans an initial route
from start to target, then walks it step by step, sampling a scenario image
at every step and checking it for a conflict. A conflict abandons the rest
of the active route, permanently blocks the point just ahead of it with a
small obstacle, and triggers a replan from the current position. The run
ends when the target is reached or the replan budget is exhausted.
"""

import glob
import json
import os
import random
import sys
from pathlib import Path
from dotenv import load_dotenv

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from industrial_use_case.planning.obstacles_2d import ObstacleMap, box_ahead_of
from industrial_use_case.planning.rrt_planner import RRTConnectPlanner
from industrial_use_case.planning.prompt_builder import build_system_prompt
from industrial_use_case.planning.llm_rrt_planner import set_llm_config, plan_full_route, RouteResult
from industrial_use_case.execution.conflict_checker import load_conflict_model, check_conflict, ConflictResult, CLASS_NAMES
from industrial_use_case.execution.session import DemoSession

# ── Fixed scenario: one rectangular room with two shelving obstacles ──────
ROOM_BOUNDS = (8.0, 6.0)
START = (0.5, 3.0)
TARGET = (7.5, 3.0)
OBSTACLES = [
    (2.0, 0.0, 3.0, 3.5),
    (4.0, 2.5, 5.0, 6.0),
]

# ── Planner / execution budgets ─────────────────────────────────────────────
STEP_SIZE = 0.5
COLLISION_RESOLUTION = 0.1
MAX_SAMPLES = 5000
MAX_LLM_RETRIES = 6
MAX_TOTAL_REPLANS = 5
SEED = 100

# ── Conflict-triggered obstacle: blocks the spot just past a detected object,
# never the object's own position (see box_ahead_of's docstring) ───────────
CONFLICT_OBSTACLE_SIZE = 0.5
CONFLICT_OBSTACLE_DISTANCE = STEP_SIZE

LOCO_VAL_IMAGES = REPO_ROOT / "data" / "LOCO" / "images" / "val"
LOCO_CHECKPOINT = REPO_ROOT / "models" / "finetuned_loco" / "best.pt"
OUTPUT_PATH = REPO_ROOT / "results" / "industrial_use_case" / "session.json"


def _sample_image(image_paths: list, rng: random.Random):
    path = rng.choice(image_paths)
    return path, cv2.imread(path)


def _route_to_dict(route: RouteResult) -> dict:
    return {
        "path": [list(p) for p in route.path],
        "reached": route.reached,
        "llm_calls": route.llm_calls,
        "conversation_log": route.conversation_log,
    }


def _conflict_to_dict(conflict: ConflictResult) -> dict:
    return {
        "triggered": conflict.triggered,
        "target_box": list(conflict.target_box) if conflict.target_box is not None else None,
        "target_class": conflict.target_class,
        "target_class_probs": (
            conflict.target_class_probs.tolist() if conflict.target_class_probs is not None else None
        ),
        "confidence": conflict.confidence,
        "relative_area": conflict.relative_area,
    }


def _session_to_dict(session: DemoSession, obstacle_map: ObstacleMap) -> dict:
    return {
        "start": list(session.start),
        "target": list(session.target),
        "room_bounds": list(ROOM_BOUNDS),
        "obstacles": [list(o) for o in OBSTACLES],
        "conflict_boxes": [list(b) for b in obstacle_map.boxes[len(OBSTACLES):]],
        "max_total_replans": session.max_total_replans,
        "final_position": list(session.current_position),
        "num_replans": session.num_replans,
        "total_steps": session.step_counter,
        "finished": session.finished,
        "reached_target": session.reached_target,
        "route_history": [
            {"trigger": e.trigger, "timestamp": e.timestamp, "route": _route_to_dict(e.route)}
            for e in session.route_history
        ],
        "conflict_log": [
            {
                "step_index": e.step_index,
                "position": list(e.position),
                "image_path": e.image_path,
                "timestamp": e.timestamp,
                "conflict": _conflict_to_dict(e.conflict),
            }
            for e in session.conflict_log
        ],
    }


def main() -> None:
    rng = random.Random(SEED)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set the GROQ_API_KEY environment variable before running the demo.")
    set_llm_config(api_key=api_key, model="openai/gpt-oss-120b", temperature=0.4, max_tokens=2048)

    obstacle_map = ObstacleMap(OBSTACLES)
    planner = RRTConnectPlanner(
        obstacle_map, ROOM_BOUNDS, step_size=STEP_SIZE, collision_resolution=COLLISION_RESOLUTION,
        max_samples=MAX_SAMPLES, rng=random.Random(SEED),
    )
    system_prompt = build_system_prompt(ROOM_BOUNDS, obstacle_map.boxes, STEP_SIZE)

    model_dense, model_prep, device = load_conflict_model(str(LOCO_CHECKPOINT))

    image_paths = sorted(glob.glob(str(LOCO_VAL_IMAGES / "*.jpg")))
    if not image_paths:
        raise RuntimeError(f"No images found under {LOCO_VAL_IMAGES}")

    session = DemoSession(start=START, target=TARGET, max_total_replans=MAX_TOTAL_REPLANS,
                           current_position=START)

    route = plan_full_route(system_prompt, planner, START, TARGET, max_llm_retries=MAX_LLM_RETRIES)
    session.record_replan(route, trigger="initial")

    while not session.finished:
        route = session.active_route
        remaining_path = route.path[1:]
        conflict_triggered = False

        for i, position in enumerate(remaining_path):
            step_idx = session.next_step_index()
            session.record_step(position)

            image_path, image_bgr = _sample_image(image_paths, rng)
            conflict = check_conflict(model_dense, model_prep, image_bgr, device=device)

            if conflict.triggered:
                session.record_conflict(step_idx, position, image_path, conflict)
                conflict_triggered = True
                previous_position = route.path[i]
                class_name = CLASS_NAMES.get(conflict.target_class, "an unrecognized object")
                print(f"Step {step_idx}: conflict ({class_name}) at {position} ({image_path})")
                break

        if conflict_triggered:
            fallback_direction = (TARGET[0] - position[0], TARGET[1] - position[1])
            box = box_ahead_of(position, previous_position, CONFLICT_OBSTACLE_SIZE,
                                CONFLICT_OBSTACLE_DISTANCE, fallback_direction)
            obstacle_map.add_box(box)

            if session.budget_exhausted():
                session.finished = True
                session.reached_target = False
                break

            system_prompt = build_system_prompt(ROOM_BOUNDS, obstacle_map.boxes, STEP_SIZE)
            initial_context = (
                f"A {class_name} was detected right at the robot's current position "
                f"{session.current_position}. Do not propose heading back to this exact "
                "point again -- plan a route that goes around it, continuing toward the "
                "target from here."
            )
            new_route = plan_full_route(
                system_prompt, planner, session.current_position, TARGET,
                max_llm_retries=MAX_LLM_RETRIES, initial_context=initial_context,
            )
            session.record_replan(new_route, trigger="conflict")
            continue

        session.finished = True
        session.reached_target = route.reached

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(_session_to_dict(session, obstacle_map), f, indent=2, ensure_ascii=False)

    print(f"Demo finished: reached_target={session.reached_target}, "
          f"steps={session.step_counter}, replans={session.num_replans}, "
          f"conflicts={len(session.conflict_log)}")
    print(f"Session saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
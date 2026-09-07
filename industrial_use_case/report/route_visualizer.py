"""
route_visualizer.py
=====================
Builds a two-tab, self-contained 2D visualization of one demo run: a static
view of the room, obstacles and the full walked route, and a constant-speed
animation of the robot walking it. Both figures are rendered client-side by
Plotly.js; this module only serializes the trace/layout/frame data as JSON.
"""

import json
import math
from typing import List, Tuple

from industrial_use_case.execution.conflict_checker import CLASS_NAMES

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]

ROBOT_SPEED = 0.5  # m/s -- constant-speed animation timing


def _background_traces(room_bounds, obstacles, start, target, reached) -> List[dict]:
    W, D = room_bounds
    traces = []
    for idx, (x0, y0, x1, y1) in enumerate(obstacles):
        traces.append({
            "type": "scatter", "x": [x0, x1, x1, x0, x0], "y": [y0, y0, y1, y1, y0],
            "mode": "lines", "fill": "toself", "fillcolor": "rgba(120,140,150,0.55)",
            "line": {"color": "#5c6b75", "width": 1},
            "name": "Obstacle", "showlegend": idx == 0, "hoverinfo": "skip",
        })
    traces.append({
        "type": "scatter", "x": [0, W, W, 0, 0], "y": [0, 0, D, D, 0], "mode": "lines",
        "line": {"color": "#9fb0ba", "width": 2}, "name": "Room boundary", "hoverinfo": "skip",
    })
    traces.append({
        "type": "scatter", "x": [start[0]], "y": [start[1]], "mode": "markers",
        "marker": {"size": 12, "color": "#2ecc71", "line": {"color": "white", "width": 1}},
        "name": "Start",
    })
    traces.append({
        "type": "scatter", "x": [target[0]], "y": [target[1]], "mode": "markers",
        "marker": {"size": 12, "color": "#2ecc71" if reached else "#e67e22", "symbol": "diamond",
                   "line": {"color": "white", "width": 1}},
        "name": "Target (reached)" if reached else "Target",
    })
    return traces


def _walked_path_traces(walked_path: List[Point]) -> List[dict]:
    """Full walked path colored by step index (static tab)."""
    xs = [p[0] for p in walked_path]
    ys = [p[1] for p in walked_path]
    return [
        {"type": "scatter", "x": xs, "y": ys, "mode": "lines",
         "line": {"color": "#d6249f", "width": 4}, "name": "Walked path"},
        {"type": "scatter", "x": xs, "y": ys, "mode": "markers",
         "marker": {"size": 6, "color": list(range(len(xs))), "colorscale": "Viridis",
                    "showscale": True, "colorbar": {"title": {"text": "Step"}, "len": 0.6}},
         "name": "Steps", "hovertemplate": "Step %{marker.color}<br>(%{x:.2f}, %{y:.2f})<extra></extra>"},
    ]


def _robot_frame_traces(walked_path: List[Point], up_to: int) -> List[dict]:
    """Path traversed so far plus the robot marker at the current step (animation tab)."""
    pts = walked_path[:up_to + 1]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [
        {"type": "scatter", "x": xs, "y": ys, "mode": "lines",
         "line": {"color": "#d6249f", "width": 4}, "name": "Walked path"},
        {"type": "scatter", "x": [pts[-1][0]], "y": [pts[-1][1]], "mode": "markers",
         "marker": {"size": 16, "color": "#ffffff", "line": {"color": "#00b8d9", "width": 3}}, "name": "Robot"},
    ]


def _abandoned_trace(points: List[Point], show: bool, showlegend: bool) -> dict:
    pts = points if show else []
    return {
        "type": "scatter", "x": [p[0] for p in pts], "y": [p[1] for p in pts], "mode": "lines",
        "line": {"color": "#ff8800", "width": 3, "dash": "dash"},
        "name": "Abandoned route", "showlegend": showlegend,
        "hovertemplate": "Abandoned<br>(%{x:.2f}, %{y:.2f})<extra></extra>",
    }


def _conflict_trace(marker: dict, show: bool, showlegend: bool) -> dict:
    pos = marker["position"] if show else None
    return {
        "type": "scatter", "x": [pos[0]] if pos else [], "y": [pos[1]] if pos else [],
        "mode": "markers", "marker": {"size": 12, "color": "#e74c3c", "symbol": "x"},
        "name": "Conflict", "showlegend": showlegend, "hovertemplate": marker["label"] + "<extra></extra>",
    }


def _conflict_box_trace(box: Box, show: bool, showlegend: bool) -> dict:
    """Small square obstacle inserted at a conflict -- same visual language as the
    static obstacles (filled rectangle), in yellow to distinguish it as a
    conflict-triggered addition rather than the scenario's original shelving."""
    x0, y0, x1, y1 = box
    xs = [x0, x1, x1, x0, x0] if show else []
    ys = [y0, y0, y1, y1, y0] if show else []
    return {
        "type": "scatter", "x": xs, "y": ys, "mode": "lines", "fill": "toself",
        "fillcolor": "rgba(255,221,0,0.55)", "line": {"color": "#c9a400", "width": 1},
        "name": "Conflict obstacle", "showlegend": showlegend, "hoverinfo": "skip",
    }


def _route_layout(room_bounds: Tuple[float, float], extra: dict = None) -> dict:
    W, D = room_bounds
    layout = {
        "xaxis": {"title": "X (m)", "range": [0, W], "gridcolor": "#e5e5ea", "zeroline": False},
        "yaxis": {"title": "Y (m)", "range": [0, D], "gridcolor": "#e5e5ea", "zeroline": False,
                  "scaleanchor": "x", "scaleratio": 1},
        "margin": {"l": 60, "r": 20, "t": 20, "b": 60},
        "legend": {"orientation": "h", "y": -0.2},
        "height": 520,
    }
    if extra:
        layout.update(extra)
    return layout


def _reconstruct_route(session: dict):
    """
    Rebuilds the trajectory actually walked, plus the un-walked tail of every
    route that was cut short by a conflict. route_history[i]'s path runs from
    where planning started to the target; only the prefix up to the matching
    conflict_log[i] position was ever walked before the next replan.
    """
    route_history = session["route_history"]
    conflict_log = session["conflict_log"]

    walked_path: List[Point] = []
    abandoned_segments: List[dict] = []
    conflict_markers: List[dict] = []

    for i, entry in enumerate(route_history):
        path = entry["route"]["path"]
        if i < len(conflict_log):
            cutoff = conflict_log[i]["position"]
            cutoff_idx = next(j for j, p in enumerate(path) if math.dist(p, cutoff) < 1e-6)
            segment, tail = path[:cutoff_idx + 1], path[cutoff_idx:]
        else:
            segment, tail = path, []

        walked_path.extend(segment if i == 0 else segment[1:])
        switch_frame = len(walked_path) - 1

        if i < len(conflict_log):
            conflict = conflict_log[i]["conflict"]
            class_name = CLASS_NAMES.get(conflict["target_class"], "unrecognized object")
            conflict_markers.append({
                "position": cutoff, "switch_frame": switch_frame,
                "label": f"{class_name} (conf. {conflict['confidence']:.2f})",
            })
        if len(tail) > 1:
            abandoned_segments.append({"points": tail, "switch_frame": switch_frame})

    return walked_path, abandoned_segments, conflict_markers


def _frame_duration_ms(walked_path: List[Point], speed: float = ROBOT_SPEED) -> int:
    if len(walked_path) < 2:
        return 500
    dists = [math.dist(walked_path[i], walked_path[i + 1]) for i in range(len(walked_path) - 1)]
    return max(100, int((sum(dists) / len(dists)) / speed * 1000))


def _build_static_figure(room_bounds, obstacles, start, target, reached,
                          walked_path, abandoned_segments, conflict_markers, conflict_boxes) -> dict:
    data = _background_traces(room_bounds, obstacles, start, target, reached)
    data += [_conflict_box_trace(box, True, k == 0) for k, box in enumerate(conflict_boxes)]
    data += [_abandoned_trace(seg["points"], True, k == 0) for k, seg in enumerate(abandoned_segments)]
    data += _walked_path_traces(walked_path)
    data += [_conflict_trace(m, True, k == 0) for k, m in enumerate(conflict_markers)]
    return {"data": data, "layout": _route_layout(room_bounds)}


def _build_animation_figure(room_bounds, obstacles, start, target, reached,
                             walked_path, abandoned_segments, conflict_markers, conflict_boxes) -> dict:
    background = _background_traces(room_bounds, obstacles, start, target, reached)
    n_bg = len(background)
    idx_abandoned = n_bg + 2
    idx_conflict = idx_abandoned + len(abandoned_segments)
    idx_conflict_box = idx_conflict + len(conflict_markers)

    initial = background + _robot_frame_traces(walked_path, up_to=0)
    initial += [_abandoned_trace(seg["points"], False, k == 0) for k, seg in enumerate(abandoned_segments)]
    initial += [_conflict_trace(m, False, k == 0) for k, m in enumerate(conflict_markers)]
    initial += [_conflict_box_trace(box, False, k == 0) for k, box in enumerate(conflict_boxes)]

    frames, slider_steps = [], []
    for i in range(len(walked_path)):
        frame_data = _robot_frame_traces(walked_path, up_to=i)
        frame_traces = [n_bg, n_bg + 1]
        for k, seg in enumerate(abandoned_segments):
            frame_data.append(_abandoned_trace(seg["points"], i >= seg["switch_frame"], k == 0))
            frame_traces.append(idx_abandoned + k)
        for k, m in enumerate(conflict_markers):
            frame_data.append(_conflict_trace(m, i >= m["switch_frame"], k == 0))
            frame_traces.append(idx_conflict + k)
        for k, box in enumerate(conflict_boxes):
            frame_data.append(_conflict_box_trace(box, i >= conflict_markers[k]["switch_frame"], k == 0))
            frame_traces.append(idx_conflict_box + k)
        frames.append({"name": str(i), "data": frame_data, "traces": frame_traces})
        slider_steps.append({
            "label": str(i), "method": "animate",
            "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate",
                                 "transition": {"duration": 0}}],
        })

    frame_ms = _frame_duration_ms(walked_path)
    layout = _route_layout(room_bounds, extra={
        "updatemenus": [{
            "type": "buttons", "showactive": False, "x": 0.0, "y": 1.15, "xanchor": "left", "yanchor": "top",
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": frame_ms, "redraw": True}, "fromcurrent": True,
                                  "transition": {"duration": min(100, frame_ms // 4)}}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                                    "transition": {"duration": 0}}]},
            ],
        }],
        "sliders": [{"active": 0, "x": 0.0, "len": 1.0, "y": -0.05,
                     "currentvalue": {"prefix": "Step: ", "visible": True}, "steps": slider_steps}],
    })
    return {"data": initial, "layout": layout, "frames": frames}


_FRAGMENT_TEMPLATE = """
<section class="route-viz">
  <h2>Route &amp; Execution</h2>
  <div class="tabs">
    <button type="button" class="tab-btn active" onclick="showRouteTab('route')">Route</button>
    <button type="button" class="tab-btn" onclick="showRouteTab('anim')">Animation (0.5 m/s)</button>
  </div>
  <div id="route-tab-route" class="tab-pane"></div>
  <div id="route-tab-anim" class="tab-pane" style="display:none;"></div>
</section>
<script>
(function() {{
  var routeData = {static_data};
  var routeLayout = {static_layout};
  var animData = {anim_data};
  var animLayout = {anim_layout};
  var animFrames = {anim_frames};
  window.__routePlotReady = false;
  Plotly.newPlot("route-tab-route", routeData, routeLayout, {{responsive: true}}).then(function() {{
    window.__routePlotReady = true;
  }});
  Plotly.newPlot("route-tab-anim", animData, animLayout, {{responsive: true}}).then(function(gd) {{
    Plotly.addFrames(gd, animFrames);
  }});
}})();
function showRouteTab(which) {{
  document.getElementById("route-tab-route").style.display = which === "route" ? "block" : "none";
  document.getElementById("route-tab-anim").style.display = which === "anim" ? "block" : "none";
  document.querySelectorAll(".route-viz .tab-btn").forEach(function(btn, i) {{
    btn.classList.toggle("active", (which === "route") === (i === 0));
  }});
}}
</script>
"""


def build_route_visualization_html(session: dict) -> str:
    room_bounds = tuple(session["room_bounds"])
    obstacles = [tuple(o) for o in session["obstacles"]]
    conflict_boxes = [tuple(b) for b in session["conflict_boxes"]]
    start, target, reached = session["start"], session["target"], session["reached_target"]

    walked_path, abandoned_segments, conflict_markers = _reconstruct_route(session)

    static_fig = _build_static_figure(room_bounds, obstacles, start, target, reached,
                                       walked_path, abandoned_segments, conflict_markers, conflict_boxes)
    anim_fig = _build_animation_figure(room_bounds, obstacles, start, target, reached,
                                        walked_path, abandoned_segments, conflict_markers, conflict_boxes)

    return _FRAGMENT_TEMPLATE.format(
        static_data=json.dumps(static_fig["data"]), static_layout=json.dumps(static_fig["layout"]),
        anim_data=json.dumps(anim_fig["data"]), anim_layout=json.dumps(anim_fig["layout"]),
        anim_frames=json.dumps(anim_fig["frames"]),
    )
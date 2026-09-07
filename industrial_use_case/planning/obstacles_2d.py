"""
obstacles_2d.py
================
Physical restriction layer for the industrial use case's synthetic warehouse
-- hard geometric constraints that a planned path must never cross.

ObstacleMap keeps a path OUT of solid boxes (shelving) -- axis-aligned
(x_min, y_min, x_max, y_max), stored in an R-tree for O(log n) point/segment
queries. compute_local_free_rects() describes the free footprint around a
point as a small set of axis-aligned rectangles, used to report nearby free
space when a candidate waypoint is rejected.
"""

from typing import List, Tuple, Iterable
import numpy as np
from rtree import index

Box = Tuple[float, float, float, float]


class ObstacleMap:
    def __init__(self, boxes: List[Box]):
        self.boxes = boxes
        p = index.Property()
        p.dimension = 2
        if boxes:
            self.index = index.Index(_bulk_load(boxes), interleaved=True, properties=p)
        else:
            self.index = index.Index(interleaved=True, properties=p)

    def is_point_free(self, point: Tuple[float, float]) -> bool:
        x, y = point
        return self.index.count((x, y, x, y)) == 0

    def is_segment_free(self, start, end, resolution: float) -> bool:
        for point in _points_along_segment(start, end, resolution):
            if not self.is_point_free(point):
                return False
        return True


def _points_along_segment(start, end, resolution: float) -> Iterable[Tuple[float, float]]:
    start_arr, end_arr = np.array(start, dtype=float), np.array(end, dtype=float)
    length = np.linalg.norm(end_arr - start_arr)
    if length == 0:
        yield tuple(start_arr)
        return
    n_samples = max(2, int(np.ceil(length / resolution)) + 1)
    for t in np.linspace(0.0, 1.0, n_samples):
        yield tuple(start_arr + t * (end_arr - start_arr))


def _bulk_load(boxes: List[Box]):
    for i, box in enumerate(boxes):
        yield (i, box, None)


def compute_local_free_rects(obstacle_map: "ObstacleMap", point: Tuple[float, float],
                              room_bounds: Tuple[float, float], max_rects: int = 3,
                              voxel_size: float = 1.0) -> List[dict]:
    """
    Describe the free footprint around `point` as up to `max_rects` axis-aligned
    rectangles, nearest to `point` first. Computed on demand when a waypoint is
    rejected by the physical check, so it is a faithful local description
    (recesses/bottlenecks excluded), not a room-wide over-approximation.
    """
    if obstacle_map is None or not obstacle_map.boxes:
        return []
    W, D = room_bounds
    px, py = point

    gx = np.arange(voxel_size / 2, W, voxel_size)
    gy = np.arange(voxel_size / 2, D, voxel_size)

    boxes = []
    for y in gy:
        free_row = [obstacle_map.is_point_free((x, y)) for x in gx]
        i = 0
        while i < len(free_row):
            if free_row[i]:
                i0 = i
                while i < len(free_row) and free_row[i]:
                    i += 1
                boxes.append((
                    gx[i0] - voxel_size / 2, y - voxel_size / 2,
                    gx[i - 1] + voxel_size / 2, y + voxel_size / 2,
                ))
            else:
                i += 1

    def _dist_to_box(b):
        x_min, y_min, x_max, y_max = b
        dx = max(x_min - px, 0.0, px - x_max)
        dy = max(y_min - py, 0.0, py - y_max)
        return float(np.hypot(dx, dy))

    boxes.sort(key=_dist_to_box)
    return [
        {"x_range": [round(b[0], 2), round(b[2], 2)],
         "y_range": [round(b[1], 2), round(b[3], 2)]}
        for b in boxes[:max_rects]
    ]
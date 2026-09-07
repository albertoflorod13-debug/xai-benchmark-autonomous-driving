"""
rrt_planner.py
==============
Based in https://github.com/motion-planning/rrt-algorithms

RRT-Connect path planner for the industrial use case's synthetic warehouse:
grows two trees, one from the start and one from the goal, and repeatedly
tries to connect them directly through free space. Finds a single feasible
path between two points -- there is no notion of path cost here, only
validity matters.

Every collision check goes through ObstacleMap, so any obstacle layout valid
there works here too.
"""

from typing import List, Optional, Tuple
import random

from rtree import index as rtree_index

from .obstacles_2d import ObstacleMap

Point = Tuple[float, float]


def _vertex_index_properties() -> rtree_index.Property:
    p = rtree_index.Property()
    p.dimension = 2
    return p


class _VertexIndex:
    """R-tree-backed store of a tree's vertices, for O(log n) nearest-neighbor queries."""

    def __init__(self):
        self._index = rtree_index.Index(interleaved=True, properties=_vertex_index_properties())
        self._count = 0
        self.parents = {}

    def add(self, vertex: Point, parent: Optional[Point]) -> None:
        self._index.insert(self._count, vertex + vertex, vertex)
        self._count += 1
        self.parents[vertex] = parent

    def nearest(self, point: Point) -> Point:
        return next(self._index.nearest(point + point, num_results=1, objects="raw"))

    def contains(self, vertex: Point) -> bool:
        return vertex in self.parents

    def path_to_root(self, vertex: Point) -> List[Point]:
        path = [vertex]
        while self.parents[path[-1]] is not None:
            path.append(self.parents[path[-1]])
        path.reverse()
        return path


def _steer(start: Point, target: Point, max_step: float) -> Point:
    dx, dy = target[0] - start[0], target[1] - start[1]
    dist = (dx ** 2 + dy ** 2) ** 0.5
    if dist <= max_step:
        return target
    scale = max_step / dist
    return (start[0] + dx * scale, start[1] + dy * scale)


class RRTConnectPlanner:
    """
    Reliability over speed: `max_samples` should be generous, since a failed
    connection after the budget only means no path was found within it, not
    that none exists.
    """

    def __init__(self, obstacle_map: ObstacleMap, room_bounds: Tuple[float, float],
                 step_size: float, collision_resolution: float, max_samples: int = 5000,
                 rng: Optional[random.Random] = None):
        self.obstacle_map = obstacle_map
        self.room_bounds = room_bounds
        self.step_size = step_size
        self.collision_resolution = collision_resolution
        self.max_samples = max_samples
        self.rng = rng or random.Random()

    def _sample_free(self) -> Point:
        W, D = self.room_bounds
        while True:
            point = (self.rng.uniform(0.0, W), self.rng.uniform(0.0, D))
            if self.obstacle_map.is_point_free(point):
                return point

    def _extend(self, tree: _VertexIndex, target: Point) -> Tuple[Optional[Point], bool]:
        nearest = tree.nearest(target)
        new_point = _steer(nearest, target, self.step_size)
        if tree.contains(new_point) or not self.obstacle_map.is_segment_free(
                nearest, new_point, self.collision_resolution):
            return None, False
        tree.add(new_point, nearest)
        return new_point, new_point == target

    def _connect(self, tree: _VertexIndex, target: Point) -> Tuple[Optional[Point], bool]:
        last_added = None
        while True:
            new_point, reached = self._extend(tree, target)
            if new_point is None:
                return last_added, False
            last_added = new_point
            if reached:
                return last_added, True

    def plan(self, start: Point, goal: Point) -> Optional[List[Point]]:
        tree_start, tree_goal = _VertexIndex(), _VertexIndex()
        tree_start.add(start, None)
        tree_goal.add(goal, None)
        grow_from_start = True

        for _ in range(self.max_samples):
            sample = self._sample_free()
            growing, other = (tree_start, tree_goal) if grow_from_start else (tree_goal, tree_start)
            new_point, _ = self._extend(growing, sample)
            if new_point is not None:
                connected_point, reached = self._connect(other, new_point)
                if reached:
                    path_from_growing = growing.path_to_root(new_point)
                    path_from_other = other.path_to_root(connected_point)
                    if grow_from_start:
                        path_from_other.reverse()
                        return path_from_growing + path_from_other[1:]
                    path_from_growing.reverse()
                    return path_from_other + path_from_growing[1:]
            grow_from_start = not grow_from_start

        return None
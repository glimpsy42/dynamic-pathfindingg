"""
Agent module: pathfinding agent with optional dynamic re-planning.
Runs search algorithm and can re-plan from current position when path is blocked.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

from grid import Grid
from node import Node
from search import ALGORITHMS, HEURISTICS


class Agent:
    """
    Agent that plans a path on the grid and can move along it.
    Supports dynamic mode: re-plan from current position when blocked.
    """

    def __init__(self, grid: Grid):
        self.grid = grid
        self.path: List[Node] = []
        self.current_index: int = 0
        self.current_position: Tuple[int, int] = grid.get_start()
        # Metrics
        self.nodes_visited: int = 0
        self.path_cost: float = 0.0
        self.execution_time_ms: float = 0.0
        self._frontier_nodes: List[Node] = []
        self._visited_nodes: List[Node] = []

    def _run_search_sync(
        self,
        algorithm: str,
        heuristic: str,
    ) -> Optional[List[Node]]:
        """Run search to completion and return path; update metrics."""
        if algorithm not in ALGORITHMS or heuristic not in HEURISTICS:
            return None
        algo_fn = ALGORITHMS[algorithm]
        h_fn = HEURISTICS[heuristic]
        self._frontier_nodes = []
        self._visited_nodes = []
        start_time = time.perf_counter()
        gen = algo_fn(self.grid, h_fn)
        path = None
        try:
            while True:
                event, node = gen.send(None)
                if event == "frontier":
                    self._frontier_nodes.append(node)
                else:
                    self._visited_nodes.append(node)
        except StopIteration as e:
            path = e.value
        self.execution_time_ms = (time.perf_counter() - start_time) * 1000
        self.nodes_visited = len(self._visited_nodes)
        if path:
            self.path_cost = path[-1].g if path else 0.0
        return path

    def plan(self, algorithm: str = "astar", heuristic: str = "manhattan") -> bool:
        """
        Plan path from current position to goal. Returns True if path found.
        """
        path = self._run_search_sync(algorithm, heuristic)
        self.path = path or []
        self.current_index = 0
        if self.path:
            self.current_position = self.path[0].position()
        return len(self.path) > 0

    def get_path(self) -> List[Node]:
        return self.path

    def get_frontier_nodes(self) -> List[Node]:
        return self._frontier_nodes

    def get_visited_nodes(self) -> List[Node]:
        return self._visited_nodes

    def get_metrics(self) -> Dict[str, float]:
        return {
            "nodes_visited": self.nodes_visited,
            "path_cost": self.path_cost,
            "execution_time_ms": self.execution_time_ms,
        }

    def move_one_step(self) -> Optional[Tuple[int, int]]:
        """Advance agent one step along path. Returns new (row, col) or None."""
        if self.current_index >= len(self.path) - 1:
            return None
        self.current_index += 1
        self.current_position = self.path[self.current_index].position()
        return self.current_position

    def is_at_goal(self) -> bool:
        return self.current_position == self.grid.get_goal()

    def is_path_blocked(self) -> bool:
        """Check if next step on path is now blocked by an obstacle."""
        if self.current_index >= len(self.path) - 1:
            return False
        next_node = self.path[self.current_index + 1]
        return not (next_node.walkable)

    def replan_from_current(
        self, algorithm: str = "astar", heuristic: str = "manhattan"
    ) -> bool:
        """Set start to current position and plan again. Returns True if path found."""
        r, c = self.current_position
        self.grid.set_start(r, c)
        return self.plan(algorithm, heuristic)

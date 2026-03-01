"""
Grid class: grid-based environment for pathfinding.
Manages rows, columns, start, goal, and obstacles.
"""

import random
from typing import List, Optional, Tuple

from node import Node


class Grid:
    """Grid-based environment with start, goal, and obstacles."""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self._grid: List[List[Node]] = []
        self._start: Tuple[int, int] = (0, 0)
        self._goal: Tuple[int, int] = (rows - 1, cols - 1)
        self._build_grid()

    def _build_grid(self) -> None:
        """Initialize grid of walkable nodes."""
        self._grid = [
            [Node(r, c) for c in range(self.cols)]
            for r in range(self.rows)
        ]
        # Set start and goal walkable
        self.get_node(self._start[0], self._start[1]).walkable = True
        self.get_node(self._goal[0], self._goal[1]).walkable = True

    def get_node(self, row: int, col: int) -> Optional[Node]:
        """Return node at (row, col) or None if out of bounds."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self._grid[row][col]
        return None

    def set_walkable(self, row: int, col: int, walkable: bool) -> None:
        """Set obstacle state; do not change start/goal."""
        node = self.get_node(row, col)
        if node and (row, col) not in (self._start, self._goal):
            node.walkable = walkable

    def toggle_wall(self, row: int, col: int) -> None:
        """Toggle wall at (row, col); ignore start/goal."""
        if (row, col) in (self._start, self._goal):
            return
        node = self.get_node(row, col)
        if node:
            node.walkable = not node.walkable

    def get_start(self) -> Tuple[int, int]:
        return self._start

    def get_goal(self) -> Tuple[int, int]:
        return self._goal

    def set_start(self, row: int, col: int) -> None:
        if self.get_node(row, col):
            self.get_node(self._start[0], self._start[1]).walkable = True
            self._start = (row, col)
            self.get_node(row, col).walkable = True

    def set_goal(self, row: int, col: int) -> None:
        if self.get_node(row, col):
            self.get_node(self._goal[0], self._goal[1]).walkable = True
            self._goal = (row, col)
            self.get_node(row, col).walkable = True

    def neighbors(self, node: Node) -> List[Node]:
        """Return list of walkable neighbor nodes (4-connected)."""
        r, c = node.row, node.col
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        out = []
        for nr, nc in candidates:
            n = self.get_node(nr, nc)
            if n and n.walkable:
                out.append(n)
        return out

    def generate_random_maze(self, obstacle_density: float = 0.3) -> None:
        """Fill grid with random obstacles; keep start and goal clear."""
        self._build_grid()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in (self._start, self._goal):
                    continue
                if random.random() < obstacle_density:
                    self._grid[r][c].walkable = False

    def reset_costs(self) -> None:
        """Reset g, h, parent on all nodes for a new search."""
        for row in self._grid:
            for node in row:
                node.g = 0.0
                node.h = 0.0
                node.parent = None

"""
Node class for grid-based pathfinding.
Represents a single cell in the grid with position and state.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Node:
    """A single cell in the pathfinding grid."""

    row: int
    col: int
    walkable: bool = True
    g: float = 0.0  # cost from start
    h: float = 0.0  # heuristic to goal
    parent: Optional["Node"] = None

    @property
    def f(self) -> float:
        """Total cost f(n) = g(n) + h(n) for A*."""
        return self.g + self.h

    def position(self) -> Tuple[int, int]:
        """Return (row, col) as tuple."""
        return (self.row, self.col)

    def __hash__(self) -> int:
        return hash((self.row, self.col))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.row == other.row and self.col == other.col

    def __lt__(self, other: "Node") -> bool:
        """For priority queue tie-breaking."""
        return self.f < other.f

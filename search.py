"""
Search module: pathfinding algorithms (Greedy Best-First, A*) and heuristics.
"""

import heapq
import math
from typing import Callable, Generator, List, Optional, Tuple

from grid import Grid
from node import Node


# Heuristic type: (current node, goal row, goal col) -> float
HeuristicFn = Callable[[Node, int, int], float]


def manhattan(node: Node, goal_row: int, goal_col: int) -> float:
    """Manhattan distance heuristic: |dr| + |dc|."""
    return abs(node.row - goal_row) + abs(node.col - goal_col)


def euclidean(node: Node, goal_row: int, goal_col: int) -> float:
    """Euclidean distance heuristic: sqrt(dr^2 + dc^2)."""
    return math.sqrt((node.row - goal_row) ** 2 + (node.col - goal_col) ** 2)


HEURISTICS = {"manhattan": manhattan, "euclidean": euclidean}


def _reconstruct_path(node: Optional[Node]) -> List[Node]:
    """Build path from goal node back to start via parent pointers."""
    path = []
    while node:
        path.append(node)
        node = node.parent
    path.reverse()
    return path


def greedy_best_first(
    grid: Grid,
    heuristic: HeuristicFn,
) -> Generator[Tuple[str, Node], None, Optional[List[Node]]]:
    """
    Greedy Best-First Search: f(n) = h(n).
    Yields ("frontier", node) or ("visited", node); returns path or None.
    """
    grid.reset_costs()
    start_r, start_c = grid.get_start()
    goal_r, goal_c = grid.get_goal()
    start = grid.get_node(start_r, start_c)
    goal = grid.get_node(goal_r, goal_c)
    if not start or not goal or not start.walkable or not goal.walkable:
        return None

    # Min-heap by h(n)
    start.h = heuristic(start, goal_r, goal_c)
    frontier = [(start.h, start)]
    seen = {start.position()}
    visited_order: List[Node] = []

    while frontier:
        _, current = heapq.heappop(frontier)
        visited_order.append(current)
        yield ("visited", current)

        if current == goal:
            return _reconstruct_path(current)

        for neighbor in grid.neighbors(current):
            pos = neighbor.position()
            if pos in seen:
                continue
            seen.add(pos)
            neighbor.h = heuristic(neighbor, goal_r, goal_c)
            neighbor.parent = current
            heapq.heappush(frontier, (neighbor.h, neighbor))
            yield ("frontier", neighbor)

    return None


def a_star(
    grid: Grid,
    heuristic: HeuristicFn,
) -> Generator[Tuple[str, Node], None, Optional[List[Node]]]:
    """
    A* Search: f(n) = g(n) + h(n).
    Yields ("frontier", node) or ("visited", node); returns path or None.
    """
    grid.reset_costs()
    start_r, start_c = grid.get_start()
    goal_r, goal_c = grid.get_goal()
    start = grid.get_node(start_r, start_c)
    goal = grid.get_node(goal_r, goal_c)
    if not start or not goal or not start.walkable or not goal.walkable:
        return None

    start.g = 0
    start.h = heuristic(start, goal_r, goal_c)
    # Min-heap by f(n), then g for tie-break
    frontier = [(start.f, start.g, start)]
    seen = {start.position(): start.g}
    visited_order: List[Node] = []

    while frontier:
        _, _, current = heapq.heappop(frontier)
        visited_order.append(current)
        yield ("visited", current)

        if current == goal:
            return _reconstruct_path(current)

        for neighbor in grid.neighbors(current):
            step_cost = 1.0
            new_g = current.g + step_cost
            pos = neighbor.position()
            if pos in seen and seen[pos] <= new_g:
                continue
            seen[pos] = new_g
            neighbor.g = new_g
            neighbor.h = heuristic(neighbor, goal_r, goal_c)
            neighbor.parent = current
            heapq.heappush(frontier, (neighbor.f, neighbor.g, neighbor))
            yield ("frontier", neighbor)

    return None


ALGORITHMS = {"greedy": greedy_best_first, "astar": a_star}

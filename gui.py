"""
GUI module: Pygame visualization for the pathfinding grid.
- Draws grid, start (green), goal (red), walls, frontier (yellow), visited (red/blue), path (green).
- Interactive editor: click to add/remove walls.
- Buttons: Run, Pause, Restart; algorithm (Greedy BFS, A*); heuristic (Manhattan, Euclidean).
"""

import pygame
import sys
from typing import Dict, List, Optional, Set, Tuple

from agent import Agent
from grid import Grid
from node import Node


# Colors (RGB)
COLOR_BG = (28, 28, 36)
COLOR_GRID = (55, 55, 68)
COLOR_START = (50, 205, 50)
COLOR_GOAL = (220, 50, 50)
COLOR_WALL = (50, 50, 60)
COLOR_FRONTIER = (255, 255, 0)
COLOR_VISITED = (255, 100, 100)
COLOR_PATH = (0, 200, 100)
COLOR_AGENT = (0, 255, 200)
COLOR_TEXT = (240, 240, 245)
COLOR_DASH = (38, 38, 48)
COLOR_BTN = (65, 70, 90)
COLOR_BTN_HOVER = (85, 92, 118)
COLOR_BTN_ACTIVE = (95, 180, 120)
COLOR_BTN_BORDER = (100, 105, 130)
COLOR_CURRENT_LABEL = (180, 220, 255)


def _draw_button(
    screen: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    font: pygame.font.Font,
    active: bool = False,
    hover: bool = False,
) -> None:
    """Draw a rounded-rect button with label."""
    color = COLOR_BTN_ACTIVE if active else (COLOR_BTN_HOVER if hover else COLOR_BTN)
    pygame.draw.rect(screen, color, rect, border_radius=6)
    pygame.draw.rect(screen, COLOR_BTN_BORDER, rect, 1, border_radius=6)
    text = font.render(label, True, COLOR_TEXT)
    tr = text.get_rect(center=rect.center)
    screen.blit(text, tr)


class PathfindingGUI:
    """Pygame window for grid visualization and controls."""

    def __init__(
        self,
        rows: int = 15,
        cols: int = 20,
        cell_size: int = 32,
        obstacle_density: float = 0.25,
    ):
        pygame.init()
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.obstacle_density = obstacle_density
        self.grid = Grid(rows, cols)
        self.agent = Agent(self.grid)
        self.dashboard_height = 100
        self.width = cols * cell_size
        self.height = rows * cell_size + self.dashboard_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dynamic Pathfinding Agent")
        self.clock = pygame.time.Clock()
        # State
        self.editing = True
        self.paused = False
        self.algorithm = "astar"
        self.heuristic = "manhattan"
        self.path: List[Node] = []
        self.frontier_set: Set[Tuple[int, int]] = set()
        self.visited_set: Set[Tuple[int, int]] = set()
        self.dynamic_mode = False
        self.font = pygame.font.Font(None, 22)
        self.font_title = pygame.font.Font(None, 26)
        # Button layout (rects for click detection)
        self._buttons: List[Tuple[pygame.Rect, str, str]] = []  # (rect, id, label)
        self._update_button_rects()

    def _cell_rect(self, row: int, col: int) -> pygame.Rect:
        """Screen rect for cell (row, col); grid starts below dashboard."""
        x = col * self.cell_size
        y = self.dashboard_height + row * self.cell_size
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    def _cell_at_pos(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        """Return (row, col) for pixel (px, py) or None."""
        if py < self.dashboard_height:
            return None
        row = (py - self.dashboard_height) // self.cell_size
        col = px // self.cell_size
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)
        return None

    def _update_button_rects(self) -> None:
        """Define button rects (call when width changes)."""
        x, y = 8, 8
        bh = 28
        self._buttons = []
        # Row 1: Run, Pause, Restart, Random maze
        for w, bid, label in [(52, "run", "Run"), (52, "pause", "Pause"), (58, "restart", "Restart"), (92, "maze", "Random Maze")]:
            r = pygame.Rect(x, y, w, bh)
            self._buttons.append((r, bid, label))
            x += w + 6
        x += 12
        # Algorithm buttons
        for w, bid, label in [(88, "algo_greedy", "Greedy BFS"), (36, "algo_astar", "A*")]:
            r = pygame.Rect(x, y, w, bh)
            self._buttons.append((r, bid, label))
            x += w + 6
        x += 10
        # Current search label (display only — we draw separately)
        # Row 2: Heuristic + metrics
        y2 = 42
        x2 = 8
        for w, bid, label in [(78, "heur_manhattan", "Manhattan"), (72, "heur_euclidean", "Euclidean")]:
            r = pygame.Rect(x2, y2, w, bh)
            self._buttons.append((r, bid, label))
            x2 += w + 6

    def _button_at(self, pos: Tuple[int, int]) -> Optional[str]:
        """Return button id if pos is inside a button, else None."""
        for rect, bid, _ in self._buttons:
            if rect.collidepoint(pos):
                return bid
        return None

    def _handle_button(self, bid: str) -> None:
        """Handle button click by id."""
        if bid == "run":
            self.editing = False
            self.paused = False
            self._run_search_animation()
        elif bid == "pause":
            self.paused = not self.paused
        elif bid == "restart":
            self._do_restart()
        elif bid == "algo_greedy":
            self.algorithm = "greedy"
        elif bid == "algo_astar":
            self.algorithm = "astar"
        elif bid == "heur_manhattan":
            self.heuristic = "manhattan"
        elif bid == "heur_euclidean":
            self.heuristic = "euclidean"
        elif bid == "maze":
            self.grid.generate_random_maze(self.obstacle_density)
            self.agent = Agent(self.grid)
            self.path = []
            self.frontier_set = set()
            self.visited_set = set()
            self.paused = False

    def _do_restart(self) -> None:
        """Reset grid view and agent to start; keep walls and algorithm."""
        self.grid.reset_costs()
        sr, sc = self.grid.get_start()
        self.agent = Agent(self.grid)
        self.path = []
        self.frontier_set = set()
        self.visited_set = set()
        self.paused = False

    def _draw_cell(self, row: int, col: int, color: Tuple[int, int, int]) -> None:
        r = self._cell_rect(row, col)
        pygame.draw.rect(self.screen, color, r)
        pygame.draw.rect(self.screen, COLOR_GRID, r, 1)

    def _draw_grid(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                node = self.grid.get_node(r, c)
                if not node.walkable:
                    self._draw_cell(r, c, COLOR_WALL)
                elif (r, c) in self.visited_set:
                    self._draw_cell(r, c, COLOR_VISITED)
                elif (r, c) in self.frontier_set:
                    self._draw_cell(r, c, COLOR_FRONTIER)
                else:
                    self._draw_cell(r, c, COLOR_BG)
        # Path on top
        path_set = {n.position() for n in self.path}
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in path_set:
                    self._draw_cell(r, c, COLOR_PATH)
        # Start and goal
        sr, sc = self.grid.get_start()
        gr, gc = self.grid.get_goal()
        self._draw_cell(sr, sc, COLOR_START)
        self._draw_cell(gr, gc, COLOR_GOAL)
        # Agent
        ar, ac = self.agent.current_position
        self._draw_cell(ar, ac, COLOR_AGENT)

    def _draw_dashboard(self, mouse_pos: Tuple[int, int]) -> None:
        """Draw dashboard: buttons, current search label, metrics."""
        surf = pygame.Surface((self.width, self.dashboard_height))
        surf.fill(COLOR_DASH)
        self.screen.blit(surf, (0, 0))
        # Separator line below dashboard
        pygame.draw.line(self.screen, COLOR_GRID, (0, self.dashboard_height), (self.width, self.dashboard_height), 1)

        # Draw buttons with hover/active state
        for rect, bid, label in self._buttons:
            active = (
                (bid == "algo_greedy" and self.algorithm == "greedy")
                or (bid == "algo_astar" and self.algorithm == "astar")
                or (bid == "heur_manhattan" and self.heuristic == "manhattan")
                or (bid == "heur_euclidean" and self.heuristic == "euclidean")
            )
            hover = rect.collidepoint(mouse_pos)
            _draw_button(self.screen, rect, label, self.font, active=active, hover=hover)

        # Current search label (right side of first row)
        algo_display = "A*" if self.algorithm == "astar" else "Greedy BFS"
        current_text = self.font_title.render(f"Current search: {algo_display}", True, COLOR_CURRENT_LABEL)
        self.screen.blit(current_text, (self.width - current_text.get_width() - 12, 14))
        # Heuristic subtitle
        h_text = self.font.render(f"Heuristic: {self.heuristic}", True, (150, 160, 180))
        self.screen.blit(h_text, (self.width - h_text.get_width() - 12, 38))

        # Metrics (bottom row of dashboard)
        m = self.agent.get_metrics()
        metrics_text = self.font.render(
            f"Nodes: {m['nodes_visited']}  |  Path cost: {m['path_cost']:.1f}  |  Time: {m['execution_time_ms']:.1f} ms",
            True, COLOR_TEXT
        )
        self.screen.blit(metrics_text, (8, self.dashboard_height - 22))

    def _run_search_animation(self) -> None:
        """Run search step-by-step and redraw (stub: run sync and show result)."""
        self.grid.reset_costs()
        self.frontier_set = set()
        self.visited_set = set()
        self.path = []
        self.agent.plan(self.algorithm, self.heuristic)
        for n in self.agent.get_frontier_nodes():
            self.frontier_set.add(n.position())
        for n in self.agent.get_visited_nodes():
            self.visited_set.add(n.position())
        self.path = self.agent.get_path()

    def run(self) -> None:
        """Main loop: handle events and redraw."""
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    bid = self._button_at(event.pos)
                    if bid:
                        self._handle_button(bid)
                    else:
                        cell = self._cell_at_pos(*event.pos)
                        if cell and self.editing:
                            self.grid.toggle_wall(cell[0], cell[1])
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_g:
                        self.grid.generate_random_maze(self.obstacle_density)
                        self.agent = Agent(self.grid)
                        self.path = []
                        self.frontier_set = set()
                        self.visited_set = set()
                    elif event.key == pygame.K_SPACE:
                        self.editing = False
                        self._run_search_animation()
                    elif event.key == pygame.K_e:
                        self.editing = not self.editing

            self.screen.fill(COLOR_BG)
            self._draw_grid()
            self._draw_dashboard(mouse_pos)
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()
        sys.exit(0)


def main() -> None:
    gui = PathfindingGUI(rows=15, cols=20, obstacle_density=0.25)
    gui.run()


if __name__ == "__main__":
    main()

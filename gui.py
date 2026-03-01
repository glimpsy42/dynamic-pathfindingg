"""
gui module - pygame visualisation for the pathfinding grid
draws grid, buttons, handles clicks and animates the search + walking
- frontier nodes = yellow
- visited nodes = blue
- final path = green
- shows real-time metrics in the dashboard
"""

import pygame
import sys
import random
import time
from typing import List, Optional, Set, Tuple

from agent import Agent
from grid import Grid
from node import Node
from search import ALGORITHMS, HEURISTICS


# -- colors --
COLOR_BG = (28, 28, 36)
COLOR_GRID = (55, 55, 68)
COLOR_START = (50, 205, 50)       # green
COLOR_GOAL = (220, 50, 50)        # red
COLOR_WALL = (50, 50, 60)
COLOR_FRONTIER = (255, 255, 0)    # yellow for frontier
COLOR_VISITED = (100, 100, 255)   # blue for visited nodes
COLOR_PATH = (0, 200, 100)        # green for final path
COLOR_AGENT = (0, 255, 200)       # cyan for agent
COLOR_TEXT = (240, 240, 245)
COLOR_DASH = (38, 38, 48)
COLOR_BTN = (65, 70, 90)
COLOR_BTN_HOVER = (85, 92, 118)
COLOR_BTN_ACTIVE = (95, 180, 120)
COLOR_BTN_BORDER = (100, 105, 130)
COLOR_LABEL = (180, 220, 255)


def _draw_button(screen, rect, label, font, active=False, hover=False):
    """draw a rounded button on screen"""
    color = COLOR_BTN_ACTIVE if active else (COLOR_BTN_HOVER if hover else COLOR_BTN)
    pygame.draw.rect(screen, color, rect, border_radius=6)
    pygame.draw.rect(screen, COLOR_BTN_BORDER, rect, 1, border_radius=6)
    text = font.render(label, True, COLOR_TEXT)
    tr = text.get_rect(center=rect.center)
    screen.blit(text, tr)


class PathfindingGUI:
    """main gui class - handles the pygame window, drawing and user interaction"""

    def __init__(self, rows=15, cols=20, cell_size=32, obstacle_density=0.25):
        pygame.init()
        self.rows = rows
        self.cols = cols
        self.obstacle_density = obstacle_density
        self.dashboard_height = 110
        # auto-size cells so window doesnt get too big
        max_gw, max_gh = 900, 650
        cw = max_gw // cols
        ch = max_gh // rows
        self.cell_size = max(min(cell_size, cw, ch), 10)

        self.grid = Grid(rows, cols)
        self.agent = Agent(self.grid)

        self.width = max(cols * self.cell_size, 680)  # min width for buttons
        self.height = rows * self.cell_size + self.dashboard_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dynamic Pathfinding Agent")
        self.clock = pygame.time.Clock()

        # state machine: editing, searching, walking, paused_search, paused_walk, done
        self.state = "editing"
        self.algorithm = "astar"
        self.heuristic = "manhattan"
        self.dynamic_mode = False

        # animation stuff
        self._search_gen = None
        self._search_result = None
        self._anim_speed = 5       # search steps per frame
        self._walk_timer = 0
        self._walk_delay = 8       # frames between walk steps
        self._original_start = (0, 0)

        # visualisation sets
        self.path: List[Node] = []
        self.frontier_set: Set[Tuple[int, int]] = set()
        self.visited_set: Set[Tuple[int, int]] = set()

        # metrics for dashboard display
        self._nodes_visited = 0
        self._path_cost = 0.0
        self._total_exec_time = 0.0
        self._search_start_time = 0

        # fonts
        self.font = pygame.font.Font(None, 22)
        self.font_title = pygame.font.Font(None, 26)
        self.font_small = pygame.font.Font(None, 20)

        # buttons
        self._buttons = []
        self._setup_buttons()

    # --- cell helpers ---

    def _cell_rect(self, row, col):
        x = col * self.cell_size
        y = self.dashboard_height + row * self.cell_size
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    def _cell_at_pos(self, px, py):
        """get (row,col) for pixel pos or None if outside grid"""
        if py < self.dashboard_height:
            return None
        row = (py - self.dashboard_height) // self.cell_size
        col = px // self.cell_size
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)
        return None

    # --- buttons ---

    def _setup_buttons(self):
        """create all the button rects"""
        self._buttons = []
        x, y = 8, 8
        bh = 28
        # row 1: action buttons + algo select
        for w, bid, label in [
            (52, "run", "Run"),
            (55, "pause", "Pause"),
            (60, "restart", "Restart"),
            (95, "maze", "Rand Maze"),
            (78, "dynamic", "Dynamic"),
        ]:
            r = pygame.Rect(x, y, w, bh)
            self._buttons.append((r, bid, label))
            x += w + 6
        x += 10
        for w, bid, label in [
            (90, "algo_greedy", "Greedy BFS"),
            (38, "algo_astar", "A*"),
        ]:
            r = pygame.Rect(x, y, w, bh)
            self._buttons.append((r, bid, label))
            x += w + 6
        # row 2: heuristic buttons
        y2, x2 = 42, 8
        for w, bid, label in [
            (80, "heur_manhattan", "Manhattan"),
            (75, "heur_euclidean", "Euclidean"),
        ]:
            r = pygame.Rect(x2, y2, w, bh)
            self._buttons.append((r, bid, label))
            x2 += w + 6

    def _button_at(self, pos):
        for rect, bid, _ in self._buttons:
            if rect.collidepoint(pos):
                return bid
        return None

    def _handle_button(self, bid):
        """handle a button click"""
        if bid == "run":
            self._start_search()
        elif bid == "pause":
            # toggle pause for whatever state we're in
            if self.state == "searching":
                self.state = "paused_search"
            elif self.state == "paused_search":
                self.state = "searching"
            elif self.state == "walking":
                self.state = "paused_walk"
            elif self.state == "paused_walk":
                self.state = "walking"
        elif bid == "restart":
            self._do_restart()
        elif bid == "maze":
            self._do_restart()
            self.grid.generate_random_maze(self.obstacle_density)
        elif bid == "dynamic":
            self.dynamic_mode = not self.dynamic_mode
        elif bid == "algo_greedy" and self.state == "editing":
            self.algorithm = "greedy"
        elif bid == "algo_astar" and self.state == "editing":
            self.algorithm = "astar"
        elif bid == "heur_manhattan" and self.state == "editing":
            self.heuristic = "manhattan"
        elif bid == "heur_euclidean" and self.state == "editing":
            self.heuristic = "euclidean"

    # --- search & walk logic ---

    def _start_search(self):
        """begin animated search from start to goal"""
        if self.state != "editing":
            return
        self.grid.reset_costs()
        self.frontier_set = set()
        self.visited_set = set()
        self.path = []
        self._nodes_visited = 0
        self._path_cost = 0.0
        self._total_exec_time = 0.0
        self._original_start = self.grid.get_start()

        algo_fn = ALGORITHMS.get(self.algorithm)
        h_fn = HEURISTICS.get(self.heuristic)
        if not algo_fn or not h_fn:
            return
        self._search_gen = algo_fn(self.grid, h_fn)
        self._search_result = None
        self._search_start_time = time.perf_counter()
        self.state = "searching"

    def _step_search(self):
        """advance search animation by a few steps per frame"""
        if not self._search_gen:
            return
        for _ in range(self._anim_speed):
            try:
                event, node = next(self._search_gen)
                pos = node.position()
                if event == "visited":
                    self.visited_set.add(pos)
                    self.frontier_set.discard(pos)  # remove from frontier when visited
                    self._nodes_visited += 1
                elif event == "frontier":
                    if pos not in self.visited_set:
                        self.frontier_set.add(pos)
            except StopIteration as e:
                # search is done
                elapsed = (time.perf_counter() - self._search_start_time) * 1000
                self._total_exec_time = elapsed
                self._search_result = e.value
                self._search_gen = None
                if self._search_result:
                    self.path = self._search_result
                    self._path_cost = self.path[-1].g if self.path else 0
                    # setup agent to walk the found path
                    self.agent.path = self.path
                    self.agent.current_index = 0
                    self.agent.current_position = self.path[0].position()
                    self._walk_timer = 0
                    self.state = "walking"
                else:
                    # no path found :(
                    self.state = "done"
                return

    def _step_walk(self):
        """move agent one step along path, handle dynamic obstacles"""
        self._walk_timer += 1
        if self._walk_timer < self._walk_delay:
            return
        self._walk_timer = 0

        # spawn dynamic obstacles if turned on
        if self.dynamic_mode:
            self._maybe_spawn_obstacle()

        # check if remaining path got blocked by new obstacle
        if self._check_path_blocked():
            self._replan()
            return

        # move agent forward one step
        result = self.agent.move_one_step()
        if result is None or self.agent.is_at_goal():
            self.state = "done"

    def _maybe_spawn_obstacle(self):
        """randomly add a wall while agent is moving around"""
        if random.random() > 0.12:
            return  # ~12% chance per step
        # try to find a empty cell to place wall on
        for _ in range(15):
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            node = self.grid.get_node(r, c)
            if (node and node.walkable
                    and (r, c) != self.grid.get_start()
                    and (r, c) != self.grid.get_goal()
                    and (r, c) != self.agent.current_position):
                node.walkable = False
                break

    def _check_path_blocked(self):
        """see if any remaining node on the path is now a wall
        only triggers replan if obstacle is actually on current path (efficent)"""
        if not self.agent.path:
            return False
        remaining = self.agent.path[self.agent.current_index + 1:]
        for node in remaining:
            real_node = self.grid.get_node(node.row, node.col)
            if real_node and not real_node.walkable:
                return True
        return False

    def _replan(self):
        """re-plan path from where agent currently is to the goal"""
        r, c = self.agent.current_position
        self.grid.set_start(r, c)
        self.grid.reset_costs()
        # clear old visualisation stuff
        self.frontier_set = set()
        self.visited_set = set()
        self.path = []
        # run search synchronously (fast enough, dont need to animate replanning)
        found = self.agent.plan(self.algorithm, self.heuristic)
        if found:
            self.path = self.agent.get_path()
            self._path_cost = self.path[-1].g if self.path else 0
            # update vis sets from the replanning search
            for n in self.agent.get_frontier_nodes():
                self.frontier_set.add(n.position())
            for n in self.agent.get_visited_nodes():
                self.visited_set.add(n.position())
            self._total_exec_time += self.agent.execution_time_ms
            self._nodes_visited += self.agent.nodes_visited
        else:
            # no path anymore, stuck
            self.state = "done"

    def _do_restart(self):
        """reset everthing back to editing mode"""
        self.grid.set_start(self._original_start[0], self._original_start[1])
        self.grid.reset_costs()
        # also clear any walls that appeared dynamically? no, keep them
        self.agent = Agent(self.grid)
        self.path = []
        self.frontier_set = set()
        self.visited_set = set()
        self.state = "editing"
        self._search_gen = None
        self._total_exec_time = 0.0
        self._nodes_visited = 0
        self._path_cost = 0.0

    # --- drawing ---

    def _draw_cell(self, row, col, color):
        r = self._cell_rect(row, col)
        pygame.draw.rect(self.screen, color, r)
        pygame.draw.rect(self.screen, COLOR_GRID, r, 1)

    def _draw_grid(self):
        """draw all grid cells with the right colors"""
        start_pos = self.grid.get_start()
        goal_pos = self.grid.get_goal()
        path_set = {n.position() for n in self.path}

        for r in range(self.rows):
            for c in range(self.cols):
                node = self.grid.get_node(r, c)
                pos = (r, c)
                if not node.walkable:
                    self._draw_cell(r, c, COLOR_WALL)
                elif pos in path_set:
                    self._draw_cell(r, c, COLOR_PATH)
                elif pos in self.visited_set:
                    self._draw_cell(r, c, COLOR_VISITED)
                elif pos in self.frontier_set:
                    self._draw_cell(r, c, COLOR_FRONTIER)
                else:
                    self._draw_cell(r, c, COLOR_BG)

        # start and goal always drawn on top so they stay visible
        sr, sc = start_pos
        gr, gc = goal_pos
        self._draw_cell(sr, sc, COLOR_START)
        self._draw_cell(gr, gc, COLOR_GOAL)

        # draw agent marker when walking or done
        if self.state in ("walking", "done", "paused_walk"):
            ar, ac = self.agent.current_position
            self._draw_cell(ar, ac, COLOR_AGENT)

    def _draw_dashboard(self, mouse_pos):
        """draw top panel with buttons and real-time metrics"""
        # background
        surf = pygame.Surface((self.width, self.dashboard_height))
        surf.fill(COLOR_DASH)
        self.screen.blit(surf, (0, 0))
        pygame.draw.line(self.screen, COLOR_GRID,
                         (0, self.dashboard_height), (self.width, self.dashboard_height), 1)

        # draw buttons with hover/active highlighting
        for rect, bid, label in self._buttons:
            active = (
                (bid == "algo_greedy" and self.algorithm == "greedy")
                or (bid == "algo_astar" and self.algorithm == "astar")
                or (bid == "heur_manhattan" and self.heuristic == "manhattan")
                or (bid == "heur_euclidean" and self.heuristic == "euclidean")
                or (bid == "dynamic" and self.dynamic_mode)
            )
            hover = rect.collidepoint(mouse_pos)
            _draw_button(self.screen, rect, label, self.font, active=active, hover=hover)

        # algo and heuristic info on the right side
        algo_name = "A*" if self.algorithm == "astar" else "Greedy BFS"
        info = self.font_title.render(f"Algo: {algo_name}", True, COLOR_LABEL)
        self.screen.blit(info, (self.width - info.get_width() - 12, 10))

        h_txt = self.font.render(f"Heuristic: {self.heuristic}", True, (150, 160, 180))
        self.screen.blit(h_txt, (self.width - h_txt.get_width() - 12, 30))

        # state and dynamic mode indicator
        state_str = self.state.replace("_", " ").title()
        dyn_str = " [Dynamic ON]" if self.dynamic_mode else ""
        st = self.font_small.render(f"State: {state_str}{dyn_str}", True, (180, 180, 190))
        self.screen.blit(st, (self.width - st.get_width() - 12, 50))

        # metrics row at bottom of dashboard
        metrics = (
            f"Nodes: {self._nodes_visited}  |  "
            f"Path cost: {self._path_cost:.1f}  |  "
            f"Time: {self._total_exec_time:.1f} ms"
        )
        mt = self.font.render(metrics, True, COLOR_TEXT)
        self.screen.blit(mt, (8, self.dashboard_height - 24))

        # hint text when in editing mode
        if self.state == "editing":
            hint = self.font_small.render(
                "Click=wall  Shift+Click=start  Ctrl+Click=goal  Space=run",
                True, (120, 120, 140))
            self.screen.blit(hint, (8, self.dashboard_height - 42))

    # --- main loop ---

    def run(self):
        """pygame main loop - events, update, draw"""
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
                        if cell and self.state == "editing":
                            mods = pygame.key.get_mods()
                            if mods & pygame.KMOD_SHIFT:
                                # shift+click sets start position
                                self.grid.set_start(cell[0], cell[1])
                                self._original_start = cell
                                self.agent = Agent(self.grid)
                            elif mods & pygame.KMOD_CTRL:
                                # ctrl+click sets goal position
                                self.grid.set_goal(cell[0], cell[1])
                            else:
                                self.grid.toggle_wall(cell[0], cell[1])
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE and self.state == "editing":
                        self._start_search()
                    elif event.key == pygame.K_g and self.state == "editing":
                        self._do_restart()
                        self.grid.generate_random_maze(self.obstacle_density)
                    elif event.key == pygame.K_r:
                        self._do_restart()
                    elif event.key == pygame.K_d:
                        self.dynamic_mode = not self.dynamic_mode

            # update based on current state
            if self.state == "searching":
                self._step_search()
            elif self.state == "walking":
                self._step_walk()

            # draw everything
            self.screen.fill(COLOR_BG)
            self._draw_grid()
            self._draw_dashboard(mouse_pos)
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
        sys.exit(0)

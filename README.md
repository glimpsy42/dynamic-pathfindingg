# pathfinding agent project

dynamic pathfinding visualisation using pygame. implements A* and Greedy Best-First search with manhattan and euclidean heuristics.

## features
- adjustable grid size (set rows and columns at startup dialog)
- random maze generation with configurable obstacle density
- interactive wall editor (click to place/remove walls)
- shift+click to move start node, ctrl+click to move goal
- step-by-step search animation (frontier = yellow, visited = blue, path = green)
- dynamic mode - random obstacles spawn while agent walks, and agent replans if path gets blocked
- real-time metrics dashboard: nodes visited, path cost, execution time (ms)
- choose between A* and Greedy BFS algorithms
- toggle between manhattan and euclidean heuristics

## files
- main.py - entry point, run this one
- gui.py - pygame GUI and visualisation
- grid.py - grid environment
- node.py - node/cell class
- search.py - search algorithms (A*, Greedy BFS)
- agent.py - pathfinding agent with replan support

## how to run

```
pip install -r requirements.txt
python main.py
```

or just:
```
pip install pygame
python main.py
```

## controls
- **Click** on grid cells to place/remove walls
- **Shift+Click** to set start position
- **Ctrl+Click** to set goal position
- **Space** or Run button to start the search
- **G** key to generate random maze
- **D** key to toggle dynamic obstacle mode
- **R** key to restart
- **Esc** to quit

## dependencies
- python 3.7+
- pygame


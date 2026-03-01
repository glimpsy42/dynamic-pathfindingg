"""
Dynamic Pathfinding Agent - entry point.
Launches the Pygame GUI.
"""

from gui import PathfindingGUI


def main() -> None:
    gui = PathfindingGUI(rows=15, cols=20, obstacle_density=0.25)
    gui.run()


if __name__ == "__main__":
    main()

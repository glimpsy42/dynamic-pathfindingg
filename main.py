"""
main file - entry point for the pathfinding project
shows a setup dialog then launches pygame window
"""

import tkinter as tk
import sys
from gui import PathfindingGUI


def get_settings():
    """show tkinter dialog to get grid size and obstacle density from user"""
    settings = {"rows": 15, "cols": 20, "density": 0.25}

    root = tk.Tk()
    root.title("Pathfinding Setup")
    root.geometry("320x230")
    root.resizable(False, False)

    tk.Label(root, text="Grid Settings", font=("Arial", 14, "bold")).pack(pady=8)

    frame = tk.Frame(root)
    frame.pack(pady=5)

    tk.Label(frame, text="Rows:").grid(row=0, column=0, padx=5, pady=4, sticky="e")
    rows_var = tk.IntVar(value=15)
    tk.Spinbox(frame, from_=5, to=50, textvariable=rows_var, width=8).grid(row=0, column=1)

    tk.Label(frame, text="Columns:").grid(row=1, column=0, padx=5, pady=4, sticky="e")
    cols_var = tk.IntVar(value=20)
    tk.Spinbox(frame, from_=5, to=50, textvariable=cols_var, width=8).grid(row=1, column=1)

    tk.Label(frame, text="Wall density %:").grid(row=2, column=0, padx=5, pady=4, sticky="e")
    density_var = tk.DoubleVar(value=25.0)
    tk.Spinbox(frame, from_=0, to=80, textvariable=density_var, width=8, increment=5.0).grid(row=2, column=1)

    def on_start():
        settings["rows"] = rows_var.get()
        settings["cols"] = cols_var.get()
        settings["density"] = density_var.get() / 100.0
        root.destroy()

    def on_close():
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    tk.Button(root, text="Start", command=on_start, width=12, font=("Arial", 11)).pack(pady=12)

    root.mainloop()
    return settings


def main():
    settings = get_settings()
    gui = PathfindingGUI(
        rows=settings["rows"],
        cols=settings["cols"],
        obstacle_density=settings["density"],
    )
    gui.run()


if __name__ == "__main__":
    main()

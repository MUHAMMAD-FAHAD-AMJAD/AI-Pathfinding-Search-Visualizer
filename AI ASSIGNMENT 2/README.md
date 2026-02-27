# Dynamic Pathfinding Agent
## AI 2002 – Artificial Intelligence Assignment 2

| Field       | Value                        |
|-------------|------------------------------|
| **Name**    | Muhammad Fahad Amjad         |
| **Roll No** | 24F-0005                     |
| **Course**  | AI 2002 – Artificial Intelligence |

A grid-based pathfinding visualizer built with Python and Pygame.  
Implements **Greedy Best-First Search** and **A\*** with real-time dynamic obstacle re-planning.

---

## Install

```bash
pip install pygame
```

## Run

```bash
python pathfinding_agent.py
```

A startup dialog will appear — enter the grid dimensions (rows × cols) and click **START**.

---

## Controls

| Action                     | How                          |
|----------------------------|------------------------------|
| Place / remove wall        | Left click on cell           |
| Draw walls by dragging     | Hold Left click + drag       |
| Move **start** node        | Right click on empty cell    |
| Move **goal** node         | Shift + Left click           |
| Run algorithm              | **RUN** button               |
| Step manually (one step)   | **STEP** button              |
| Reset path (keep walls)    | **RESET** button             |
| Clear everything           | **CLEAR** button             |
| New random maze            | **GENERATE** button          |
| Toggle dynamic obstacles   | **DYNAMIC MODE** button      |
| Apply new grid dimensions  | Edit rows/cols → **APPLY GRID SIZE** |
| Quit                       | Esc                          |

---

## Grid Visual Legend

| Color        | Meaning                         |
|--------------|---------------------------------|
| White        | Empty cell                      |
| Black        | Wall / obstacle                 |
| Dark Green   | Start node                      |
| Red          | Goal node                       |
| Yellow       | Frontier (in priority queue)    |
| Light Blue   | Visited / expanded node         |
| Bright Green | Final optimal path              |
| Orange dot   | Agent moving along path         |
| Flash Red    | Newly spawned dynamic wall      |

---

## Algorithms

### Greedy Best-First Search (GBFS)
$$f(n) = h(n)$$

Uses **only the heuristic** — fast but **not guaranteed optimal**.  
Expands the node that looks closest to the goal at each step.

### A* Search
$$f(n) = g(n) + h(n)$$

- $g(n)$ = actual cost from start to node $n$ (each step costs `move_weight`)  
- $h(n)$ = heuristic estimate to goal  

Slower than GBFS but **always finds the optimal path** (with admissible heuristics).

---

## Heuristics

| Heuristic           | Formula                                                        |
|---------------------|----------------------------------------------------------------|
| Manhattan Distance  | $D = |r_1 - r_2| + |c_1 - c_2|$                              |
| Euclidean Distance  | $D = \sqrt{(r_1-r_2)^2 + (c_1-c_2)^2}$                      |

Movement is **4-directional** (up, down, left, right) only.

---

## Movement Weight

Located in the **Move Weight** input field on the right panel.  
Default: `1`

Each step costs `move_weight` instead of 1.

- Weight = 1 → normal unit cost  
- Weight = 2 → every step costs 2 (Path Cost doubles for equal-length roads)  
- Weight = 3 → teacher can verify Path Cost = `(path_length - 1) × weight`

---

## Dynamic Mode

When **DYNAMIC MODE** is ON and the agent is moving along the path:

- Each frame has a ~3 % chance of spawning a random wall on an empty cell.
- The new wall **flashes red/orange** briefly before turning black.
- **Efficiency rule**: if the new wall is NOT on the agent's remaining path, the agent continues without replanning.
- If the new wall **IS on the remaining path**, the agent immediately replans from its current position (not from the start).

---

## Metrics Dashboard

Displayed live in the bottom-right panel:

| Metric          | Description                                        |
|-----------------|----------------------------------------------------|
| **Nodes Visited** | Total nodes expanded during the search           |
| **Path Cost**     | Sum of step costs for the final path found       |
| **Time (ms)**     | Milliseconds from RUN click to goal discovery    |

---

## Project Structure

```
AI ASSIGNMENT 2/
├── pathfinding_agent.py   # Complete single-file application
└── README.md              # This file
```

---

## Algorithm Comparison (based on experimental findings)

| Criterion         | GBFS                        | A*                          |
|-------------------|-----------------------------|-----------------------------|
| Optimality        | ✗ Not guaranteed            | ✓ Always optimal            |
| Speed             | ✓ Fast (few nodes expanded) | Slower (explores carefully) |
| Nodes Visited     | Fewer (greedy)              | More (but safer)            |
| Suitability       | Real-time / speed priority  | Correctness priority        |
| Heuristic effect  | Heavily dependent           | Balanced with path cost     |

---

## Viva Checklist (Teacher Will Test These Live)

- [x] Switch A* ↔ GBFS and run — different expansions, same/different path
- [x] Switch Manhattan ↔ Euclidean — subtle path differences visible
- [x] Change Move Weight 1 → 3 — Path Cost in dashboard multiplies accordingly
- [x] Enable Dynamic Mode — watch agent replan when new wall hits its path
- [x] Change grid to 10×10 or 30×30 via APPLY GRID SIZE
- [x] Set density to 50% and GENERATE new maze
- [x] Draw custom walls manually, then RUN

---

## Dependencies

```
pygame >= 2.0
Python >= 3.10
```

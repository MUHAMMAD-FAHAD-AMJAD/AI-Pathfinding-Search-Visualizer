# Dynamic Pathfinding Agent

**AI 2002 – Artificial Intelligence Assignment 2**

| Field       | Value                              |
|-------------|-------------------------------------|
| **Name**    | Muhammad Fahad Amjad               |
| **Roll No** | 24F-0005                           |
| **Course**  | AI 2002 – Artificial Intelligence  |

A grid-based pathfinding visualizer built with **Python** and **Pygame**.  
Implements **Greedy Best-First Search (GBFS)** and **A\*** with real-time dynamic obstacle re-planning.

---

## Install

```bash
pip install pygame
```

## Run

```bash
python pathfinding_agent.py
```

The application launches directly into a 20×20 grid. Use the right-side panel to change grid size, algorithm, heuristic, and all other settings.

---

## Controls

| Action                     | How                                   |
|----------------------------|---------------------------------------|
| Place / remove wall        | Left-click on a cell                  |
| Draw walls by dragging     | Hold left-click and drag              |
| Move **Start** node        | Right-click on an empty cell          |
| Move **Goal** node         | Shift + Left-click on an empty cell   |
| Run algorithm              | **▶ Run** button                      |
| Step one expansion         | **⏭ Step** button                     |
| Reset path (keep walls)    | **↺ Reset** button                    |
| Clear everything           | **✕ Clear** button                    |
| Generate random maze       | **⟳ Generate** button                 |
| Toggle dynamic obstacles   | **Dynamic: On/Off** button            |
| Change grid dimensions     | Edit Rows/Cols → **Apply**            |
| Quit                       | Press Escape or close window          |

---

## Color Legend

| Color      | Meaning                              |
|------------|--------------------------------------|
| Green ▶    | Start node                           |
| Red ●      | Goal node                            |
| Dark       | Wall / obstacle                      |
| Amber      | Frontier (nodes in priority queue)   |
| Blue       | Visited / expanded node              |
| Green      | Final path                           |
| Orange dot | Agent moving along path              |
| Flash Red  | Newly spawned dynamic obstacle       |

---

## Algorithms

### Greedy Best-First Search (GBFS)

$$f(n) = h(n)$$

Uses **only the heuristic** — expands the node that looks closest to the goal.  
Fast but **not guaranteed optimal**.

### A\* Search

$$f(n) = g(n) + h(n)$$

- $g(n)$ = actual cost from start to node $n$ (each step costs `move_weight`)
- $h(n)$ = heuristic estimate to goal

Slower than GBFS but **always finds the optimal path** when using an admissible heuristic.

---

## Heuristics

| Heuristic          | Formula                                               |
|--------------------|-------------------------------------------------------|
| Manhattan Distance | $|r_1 - r_2| + |c_1 - c_2|$                          |
| Euclidean Distance | $\sqrt{(r_1 - r_2)^2 + (c_1 - c_2)^2}$              |

Both are **admissible** for 4-directional grid movement.

---

## Movement Weight

The **Weight** input in the Parameters card controls the cost per step.

- Weight = 1 → standard unit cost
- Weight = 2 → every step costs 2, so Path Cost doubles for equal-length paths
- Weight = 3 → Path Cost = (path_length − 1) × 3

This directly affects A\*'s $g(n)$ computation and can be changed live during the viva.

---

## Dynamic Mode

When **Dynamic: On** is active and the agent is traversing the path:

1. Each frame has a **~3% chance** of spawning a new wall on a random empty cell.
2. The new wall **flashes red/orange** briefly before becoming permanent.
3. **Efficiency rule**: if the new wall is NOT on the agent's remaining path, no action is taken.
4. If the new wall **IS on the remaining path**, the agent immediately **replans from its current position** (not from the start).

---

## Metrics Dashboard

Displayed live in the panel:

| Metric             | Description                                   |
|--------------------|-----------------------------------------------|
| **Nodes Visited**  | Total nodes expanded during the search        |
| **Path Cost**      | Total cost of the final path                  |
| **Time (ms)**      | Computation time from Run to goal discovery   |

---

## Project Structure

```
AI ASSIGNMENT 2/
├── pathfinding_agent.py   # Complete single-file application
├── README.md              # This file
└── .gitignore
```

---

## Algorithm Comparison

| Criterion       | GBFS                          | A\*                           |
|-----------------|-------------------------------|-------------------------------|
| Optimality      | ✗ Not guaranteed              | ✓ Guaranteed (admissible h)   |
| Speed           | ✓ Fewer nodes expanded        | Slower (explores carefully)   |
| Nodes Visited   | Fewer (greedy)                | More (but safer)              |
| Best for        | Speed-critical scenarios      | Correctness-critical tasks    |
| Heuristic role  | Dominates behavior            | Balanced with path cost       |

---

## Dependencies

```
Python >= 3.10
pygame >= 2.0
```

<div align="center">

# 🤖 AI Pathfinding Search Visualizer

### Interactive Visualization of Uninformed Search Algorithms

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.5%2B-orange.svg)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

</div>

---

## 📑 Table of Contents

- [About](#about)
- [Features](#features)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Scenarios](#scenarios)
- [Color Legend](#color-legend)
- [Technical Details](#technical-details)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

---

## 🎯 About

This project implements a **real-time visualization tool** for comparing six fundamental uninformed search algorithms in a grid-based pathfinding environment. Built with Python and Matplotlib, it provides an interactive GUI to observe how different algorithms explore the search space and find paths from a start point to a target.

**Academic Context:** AI 2002 – Artificial Intelligence (Spring 2026) - Assignment 1

---

## ✨ Features

- 🎨 **Interactive GUI** with real-time visualization
- 🔍 **6 Search Algorithms** fully implemented
- 🗺️ **3 Pre-configured Scenarios** (Best Case, Worst Case, Medium)
- ⚡ **Adjustable Speed Control** (Slow, Medium, Fast, Instant)
- 📊 **Live Statistics** (Nodes explored, Path length, Success rate)
- 🎯 **Step-by-step Animation** showing algorithm thinking process
- 📈 **Performance Comparison** across different scenarios
- 🖱️ **One-Click Execution** via RUN button

---

## 🧠 Algorithms

### 1. **BFS (Breadth-First Search)**
- **Strategy:** Explores nodes level by level using a queue
- **Guarantees:** Shortest path in unweighted graphs
- **Best For:** Finding optimal solutions in sparse grids
- **Complexity:** O(V + E)

### 2. **DFS (Depth-First Search)**
- **Strategy:** Explores as deep as possible before backtracking using a stack
- **Characteristics:** Memory efficient but may not find shortest path
- **Best For:** Deep exploration and memory-constrained environments
- **Complexity:** O(V + E)

### 3. **UCS (Uniform-Cost Search)**
- **Strategy:** Expands nodes with lowest cumulative cost using priority queue
- **Guarantees:** Optimal path when edges have different costs
- **Best For:** Weighted pathfinding (diagonal moves cost more)
- **Complexity:** O((V + E) log V)

### 4. **DLS (Depth-Limited Search)**
- **Strategy:** DFS with a maximum depth limit (set to 15)
- **Characteristics:** Prevents infinite loops in cyclic graphs
- **Best For:** Scenarios with known solution depth
- **Complexity:** O(b^d) where d is depth limit

### 5. **IDDFS (Iterative Deepening DFS)**
- **Strategy:** Runs DLS repeatedly with increasing depth limits
- **Combines:** Space efficiency of DFS + Optimality of BFS
- **Best For:** Unknown solution depth with memory constraints
- **Complexity:** O(b^d)

### 6. **Bidirectional Search**
- **Strategy:** Searches simultaneously from start and target
- **Characteristics:** Meets in the middle, significantly faster
- **Best For:** Known start and goal positions
- **Complexity:** O(b^(d/2))

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/MUHAMMAD-FAHAD-AMJAD/AI-Pathfinding-Search-Visualizer.git
cd AI-Pathfinding-Search-Visualizer
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install matplotlib directly:

```bash
pip install matplotlib
```

### Step 3: Verify Installation

```bash
python --version
python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
```

---

## 💻 Usage

### Basic Execution

```bash
python main.py
```

### Step-by-Step Guide

1. **Launch the Application**
   ```bash
   python main.py
   ```

2. **Select Algorithm** 
   - Choose from 6 algorithms via radio buttons (right panel)

3. **Choose Scenario**
   - Scenario 1: Vertical wall obstacle (moderate)
   - Scenario 2: Complex maze (worst case)
   - Scenario 3: Open space (best case)

4. **Set Animation Speed**
   - **Slow:** 0.5s delay between steps (educational)
   - **Medium:** 0.1s delay (balanced)
   - **Fast:** 0.01s delay (quick demo)
   - **Instant:** No delay (immediate result)

5. **Click 'RUN ALGORITHM' Button**
   - Watch the visualization execute in real-time
   - Observe frontier expansion (yellow cells)
   - See explored nodes (light blue cells)
   - Final path highlights in green

6. **View Results**
   - Statistics appear in overlay box
   - Terminal shows detailed output
   - Ready to test another algorithm/scenario

---

## 📁 Project Structure

```
AI-Pathfinding-Search-Visualizer/
│
├── main.py                 # Main application file (all code)
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation (this file)
├── .gitignore             # Git ignore rules
│
└── Screenshots/           # (Create this for your report)
    ├── BFS_Scenario1.png
    ├── DFS_Scenario2.png
    └── ...
```

---

## 🗺️ Scenarios

### Scenario 1: Vertical Wall Obstacle
- **Difficulty:** Moderate
- **Description:** Single vertical wall dividing the grid
- **Purpose:** Tests algorithm efficiency with simple obstacles
- **Best Algorithm:** BFS (finds shortest path around wall)

### Scenario 2: Complex Maze
- **Difficulty:** Hard (Worst Case)
- **Description:** Multiple walls creating a maze structure
- **Purpose:** Tests algorithm performance under challenging conditions
- **Best Algorithm:** Bidirectional (searches from both ends)

### Scenario 3: Open Space
- **Difficulty:** Easy (Best Case)
- **Description:** No obstacles, direct path available
- **Purpose:** Baseline performance testing
- **Best Algorithm:** All perform well; BFS/Bidirectional fastest

---

## 🎨 Color Legend

| Color | Meaning | Description |
|-------|---------|-------------|
| 🟦 Blue (S) | **Start** | Initial position |
| 🟩 Green (T) | **Target** | Goal destination |
| 🟥 Red (-1) | **Wall** | Impassable obstacle |
| 🟨 Yellow | **Frontier** | Nodes in queue/stack waiting to be explored |
| 🔵 Light Blue | **Explored** | Nodes already visited |
| 🟢 Green Line | **Path** | Final solution route |
| ⬜ White/Gray | **Empty** | Passable cell (value: 0) |

---

## 🔧 Technical Details

### Movement Order (Clockwise)
The algorithms expand neighbors in this specific order:
1. ⬆️ **Up** (North)
2. ➡️ **Right** (East)
3. ⬇️ **Down** (South)
4. ↘️ **Down-Right** (Southeast Diagonal)
5. ⬅️ **Left** (West)
6. ↖️ **Up-Left** (Northwest Diagonal)

**Note:** Only two main diagonals allowed (no Northeast or Southwest).

### Grid Representation
- **Grid Size:** 10×10 cells
- **Cell Values:**
  - `0` = Passable cell
  - `-1` = Wall/Obstacle
  - `'S'` = Start position
  - `'T'` = Target position

### Cost Function (UCS)
- **Straight moves** (Up, Down, Left, Right): Cost = 1.0
- **Diagonal moves** (Down-Right, Up-Left): Cost = 1.414 (√2)

---

## 👥 Authors

**Group Members:**
- **Muhammad Fahad Amjad** - 24F-0005
  - 🔗 GitHub: [@MUHAMMAD-FAHAD-AMJAD](https://github.com/MUHAMMAD-FAHAD-AMJAD)
  
- **Muhammad Alyan Riaz** - 24F-0783
  - 🔗 GitHub: [@MalikAlyan](https://github.com/MalikAlyan)

**Course:** AI 2002 – Artificial Intelligence

---

## 📝 Acknowledgments

- Assignment designed for AI 2002 course curriculum
- Visualization inspired by pathfinding algorithm demonstrations
- Matplotlib library for powerful Python plotting capabilities
- Academic references from AI textbooks (Russell & Norvig, etc.)

---

## 📊 Performance Comparison

### Expected Results Across Scenarios

| Algorithm | Scenario 1 | Scenario 2 | Scenario 3 | Optimal Path? |
|-----------|------------|------------|------------|---------------|
| BFS | ~46 nodes | ~78 nodes | ~10 nodes | ✅ Yes |
| DFS | ~74 nodes | ~89 nodes | ~25 nodes | ❌ No |
| UCS | ~52 nodes | ~81 nodes | ~12 nodes | ✅ Yes |
| DLS | ~35 nodes | May fail | ~15 nodes | ❌ Sometimes |
| IDDFS | ~68 nodes | ~92 nodes | ~18 nodes | ✅ Yes |
| Bidirectional | ~28 nodes | ~45 nodes | ~8 nodes | ✅ Yes |

*Note: Actual values may vary based on grid configuration*

---

## 🤝 Contributing

This is an academic project. For improvements or bug reports:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

## 📄 License

This project is developed for educational purposes as part of university coursework.

---

## 📧 Contact

For questions or feedback:
- **Email:** f240005@cfd.nu.edu.pk
- **GitHub Issues:** [Project Issues Page](https://github.com/MUHAMMAD-FAHAD-AMJAD/AI-Pathfinding-Search-Visualizer/issues)

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

Made with ❤️ for AI 2002 Assignment

</div>

"""
Dynamic Pathfinding Agent
AI 2002 - Artificial Intelligence Assignment 2
Author : Muhammad Fahad Amjad
Roll No: 24F-0005
"""

# ============================================================
# IMPORTS
# ============================================================
import pygame
import sys
import heapq
import math
import random
import time

pygame.init()
pygame.font.init()

# ============================================================
# CONSTANTS AND COLORS
# ============================================================
WINDOW_WIDTH   = 1100
WINDOW_HEIGHT  = 700
GRID_AREA_W    = 700
GRID_AREA_H    = 700
PANEL_X        = 700
PANEL_W        = 400
PANEL_H        = 700

WHITE        = (255, 255, 255)
BLACK        = (0,   0,   0  )
DARK_GREEN   = (0,   110, 0  )
RED          = (220, 20,  60 )
YELLOW       = (255, 215, 0  )
LIGHT_BLUE   = (100, 180, 230)
BRIGHT_GREEN = (0,   210, 0  )
ORANGE       = (255, 140, 0  )
GRAY         = (150, 150, 150)
LIGHT_GRAY   = (210, 210, 210)
DARK_GRAY    = (55,  55,  55 )
PANEL_BG     = (26,  26,  38 )
PANEL_BORDER = (50,  50,  80 )
BTN_COLOR    = (52,  52,  82 )
BTN_HOVER    = (72,  72,  130)
BTN_ACTIVE   = (60,  120, 210)
BTN_ON       = (40,  170, 70 )
BTN_OFF      = (170, 55,  55 )
TEXT_COLOR   = (220, 220, 220)
TITLE_COLOR  = (160, 190, 255)
METRIC_BG    = (16,  16,  26 )
METRIC_VAL   = (80,  220, 140)
FLASH_RED    = (255, 40,  40 )
GRID_LINE    = (180, 180, 180)
INPUT_BG     = (38,  38,  60 )
INPUT_BORDER = (100, 100, 160)
INPUT_ACTIVE = (80,  140, 220)

FONT_SM   = pygame.font.SysFont("segoeui",  14)
FONT_MD   = pygame.font.SysFont("segoeui",  16)
FONT_LG   = pygame.font.SysFont("segoeui",  18, bold=True)
FONT_XL   = pygame.font.SysFont("segoeui",  22, bold=True)
FONT_MONO = pygame.font.SysFont("consolas", 15)
SPEED_MAP = {"Slow": 120, "Medium": 40, "Fast": 5}

# ============================================================
# NODE CLASS
# ============================================================
class Node:
    def __init__(self, row, col, total_rows, total_cols):
        self.row        = row
        self.col        = col
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.color      = WHITE
        self.neighbors  = []
        self.g          = float("inf")
        self.h          = 0.0
        self.f          = 0.0
        self.parent     = None

    def __lt__(self, other):
        return self.f < other.f

    def is_wall(self)     -> bool: return self.color == BLACK
    def is_start(self)    -> bool: return self.color == DARK_GREEN
    def is_goal(self)     -> bool: return self.color == RED
    def is_visited(self)  -> bool: return self.color == LIGHT_BLUE
    def is_frontier(self) -> bool: return self.color == YELLOW
    def is_path(self)     -> bool: return self.color == BRIGHT_GREEN

    def make_wall(self)     -> None: self.color = BLACK
    def make_start(self)    -> None: self.color = DARK_GREEN
    def make_goal(self)     -> None: self.color = RED
    def make_visited(self)  -> None: self.color = LIGHT_BLUE
    def make_frontier(self) -> None: self.color = YELLOW
    def make_path(self)     -> None: self.color = BRIGHT_GREEN
    def reset(self)         -> None: self.color = WHITE

    def update_neighbors(self, grid):
        self.neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = self.row + dr, self.col + dc
            if 0 <= r < self.total_rows and 0 <= c < self.total_cols:
                if not grid[r][c].is_wall():
                    self.neighbors.append(grid[r][c])

    def draw(self, surface, cell_size):
        pygame.draw.rect(surface, self.color,
                         (self.col * cell_size, self.row * cell_size,
                          cell_size, cell_size))



# ============================================================
# GRID CLASS
# ============================================================
class Grid:
    def __init__(self, rows, cols):
        self.rows      = rows
        self.cols      = cols
        self.cell_size = min(GRID_AREA_W // cols, GRID_AREA_H // rows)
        self.nodes     = [[Node(r, c, rows, cols) for c in range(cols)]
                          for r in range(rows)]
        self.start = self.nodes[0][0]
        self.goal  = self.nodes[rows - 1][cols - 1]
        self.start.make_start()
        self.goal.make_goal()

    def reset_path(self):
        for row in self.nodes:
            for node in row:
                if node.is_visited() or node.is_frontier() or node.is_path():
                    node.reset()
        self.start.make_start()
        self.goal.make_goal()

    def clear(self):
        for row in self.nodes:
            for node in row:
                node.reset()
                node.g = float("inf")
                node.h = 0.0
                node.f = 0.0
                node.parent   = None
                node.neighbors = []
        self.start.make_start()
        self.goal.make_goal()

    def generate_maze(self, density):
        for _ in range(50):
            self.clear()
            for r in range(self.rows):
                for c in range(self.cols):
                    n = self.nodes[r][c]
                    if n is self.start or n is self.goal:
                        continue
                    if random.random() < density / 100:
                        n.make_wall()
            if self._path_exists():
                return
        self.clear()

    def _path_exists(self):
        visited = {self.start}
        queue   = [self.start]
        while queue:
            cur = queue.pop(0)
            if cur is self.goal:
                return True
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                r, c = cur.row + dr, cur.col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    nb = self.nodes[r][c]
                    if nb not in visited and not nb.is_wall():
                        visited.add(nb)
                        queue.append(nb)
        return False

    def update_all_neighbors(self):
        for row in self.nodes:
            for node in row:
                node.update_neighbors(self.nodes)

    def draw(self, surface):
        cs = self.cell_size
        for row in self.nodes:
            for node in row:
                node.draw(surface, cs)
        for r in range(self.rows + 1):
            pygame.draw.line(surface, GRID_LINE, (0, r*cs), (self.cols*cs, r*cs))
        for c in range(self.cols + 1):
            pygame.draw.line(surface, GRID_LINE, (c*cs, 0), (c*cs, self.rows*cs))

    def get_node_at_pixel(self, x, y):
        cs = self.cell_size
        col, row = x // cs, y // cs
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.nodes[row][col]
        return None



# ============================================================
# HEURISTICS
# ============================================================
def heuristic_manhattan(node, goal):
    return abs(node.row - goal.row) + abs(node.col - goal.col)


def heuristic_euclidean(node, goal):
    return math.sqrt((node.row - goal.row)**2 + (node.col - goal.col)**2)


# ============================================================
# PATH RECONSTRUCTION
# ============================================================
def _reconstruct_path(came_from, start, goal):
    path, node = [], goal
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path


# ============================================================
# A STAR SEARCH
# ============================================================
def astar(grid, start, goal, heuristic_fn, move_weight, metrics):
    """Generator: yields (came_from, path|None) at each step."""
    counter = 0
    open_heap = []
    came_from = {}
    g_score   = {start: 0.0}
    in_open   = {start}

    def push(node, f_val):
        nonlocal counter
        heapq.heappush(open_heap, (f_val, counter, node))
        counter += 1
        in_open.add(node)

    h0 = heuristic_fn(start, goal)
    start.h = h0
    start.f = h0
    push(start, h0)

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current not in g_score:
            yield came_from, None
            continue
        current_g = g_score[current]

        if current is goal:
            path = _reconstruct_path(came_from, start, goal)
            metrics["path_cost"]     = round(g_score[goal], 4)
            metrics["nodes_visited"] = len(g_score)
            yield came_from, path
            return

        if current is not start and current is not goal:
            current.make_visited()

        for nb in current.neighbors:
            tg = current_g + move_weight
            if tg < g_score.get(nb, float("inf")):
                came_from[nb]  = current
                g_score[nb]    = tg
                h_val          = heuristic_fn(nb, goal)
                f_val          = tg + h_val
                nb.h           = h_val
                nb.f           = f_val
                metrics["nodes_visited"] += 1
                push(nb, f_val)
                if nb is not goal:
                    nb.make_frontier()

        yield came_from, None

    yield came_from, []


if __name__ == "__main__":
    print("Stage 3 OK")



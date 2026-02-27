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


if __name__ == "__main__":
    print("Stage 1 OK")

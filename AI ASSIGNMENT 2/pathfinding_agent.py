"""
Dynamic Pathfinding Agent
=========================
AI 2002 – Artificial Intelligence Assignment 2
Author : Muhammad Fahad Amjad
Roll No: 24F-0005

Algorithms : Greedy Best-First Search (GBFS) | A* Search
Heuristics : Manhattan Distance | Euclidean Distance
GUI        : Pygame  (pip install pygame)
"""

import pygame
import sys
import heapq
import math
import random
import time

pygame.init()
pygame.font.init()

# ─────────────────────────────────────────────────────────────
# WINDOW
# ─────────────────────────────────────────────────────────────
INIT_W  = 1280
INIT_H  = 780
PANEL_W = 340

# ─────────────────────────────────────────────────────────────
# MODERN DARK PALETTE
# ─────────────────────────────────────────────────────────────
BG_DARK       = (13, 17, 23)
PANEL_BG      = (22, 27, 34)
CARD_BG       = (30, 37, 48)
CARD_BORDER   = (48, 54, 61)

GRID_BG       = (246, 248, 250)
GRID_LINE_COL = (216, 222, 228)
WALL_COLOR    = (31, 35, 40)

ACCENT_BLUE   = (31, 111, 235)
ACCENT_GREEN  = (46, 160, 67)
ACCENT_RED    = (218, 54, 51)
ACCENT_AMBER  = (210, 153, 34)
ACCENT_ORANGE = (240, 136, 62)
ACCENT_TEAL   = (63, 185, 80)
ACCENT_CYAN   = (88, 166, 255)

START_COLOR   = (46, 160, 67)
GOAL_COLOR    = (218, 54, 51)
VISITED_COLOR = (56, 139, 253)
FRONTIER_COL  = (210, 153, 34)
PATH_COLOR    = (63, 185, 80)
FLASH_RED     = (255, 50, 50)

TEXT_PRIMARY   = (230, 237, 243)
TEXT_SECONDARY = (139, 148, 158)
TEXT_MUTED     = (110, 118, 129)

BTN_DEFAULT  = (33, 38, 45)
BTN_HOVER    = (48, 54, 61)
BTN_ACTIVE   = (31, 111, 235)
BTN_GREEN    = (35, 134, 54)
BTN_GREEN_HV = (46, 160, 67)
BTN_BORDER   = (48, 54, 61)

INP_BG     = (13, 17, 23)
INP_BORDER = (48, 54, 61)
INP_FOCUS  = (88, 166, 255)

METRIC_VAL = (88, 166, 255)

# Aliases used by Node
WHITE  = GRID_BG
BLACK  = WALL_COLOR
ORANGE = ACCENT_ORANGE

# ─────────────────────────────────────────────────────────────
# FONTS
# ─────────────────────────────────────────────────────────────
FONT_XS   = pygame.font.SysFont("segoeui", 11)
FONT_SM   = pygame.font.SysFont("segoeui", 13)
FONT_MD   = pygame.font.SysFont("segoeui", 14)
FONT_LG   = pygame.font.SysFont("segoeui", 15, bold=True)
FONT_XL   = pygame.font.SysFont("segoeui", 18, bold=True)
FONT_MONO = pygame.font.SysFont("consolas", 13)
FONT_ICON = pygame.font.SysFont("segoeui", 11, bold=True)

SPEED_MAP = {"Slow": 120, "Medium": 40, "Fast": 5}


# ══════════════════════════════════════════════════════════════
# NODE
# ══════════════════════════════════════════════════════════════
class Node:
    __slots__ = ("row", "col", "total_rows", "total_cols",
                 "color", "neighbors", "g", "h", "f", "parent")

    def __init__(self, row: int, col: int, total_rows: int, total_cols: int):
        self.row        = row
        self.col        = col
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.color      = GRID_BG
        self.neighbors: list["Node"] = []
        self.g: float   = float("inf")
        self.h: float   = 0.0
        self.f: float   = 0.0
        self.parent     = None

    def __lt__(self, other: "Node") -> bool:
        return self.f < other.f

    def is_wall(self)     -> bool: return self.color == WALL_COLOR
    def is_start(self)    -> bool: return self.color == START_COLOR
    def is_goal(self)     -> bool: return self.color == GOAL_COLOR
    def is_visited(self)  -> bool: return self.color == VISITED_COLOR
    def is_frontier(self) -> bool: return self.color == FRONTIER_COL
    def is_path(self)     -> bool: return self.color == PATH_COLOR

    def make_wall(self)     -> None: self.color = WALL_COLOR
    def make_start(self)    -> None: self.color = START_COLOR
    def make_goal(self)     -> None: self.color = GOAL_COLOR
    def make_visited(self)  -> None: self.color = VISITED_COLOR
    def make_frontier(self) -> None: self.color = FRONTIER_COL
    def make_path(self)     -> None: self.color = PATH_COLOR
    def reset(self)         -> None: self.color = GRID_BG

    def update_neighbors(self, grid: list[list["Node"]]) -> None:
        self.neighbors = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = self.row + dr, self.col + dc
            if 0 <= r < self.total_rows and 0 <= c < self.total_cols:
                if not grid[r][c].is_wall():
                    self.neighbors.append(grid[r][c])

    def draw(self, surface: pygame.Surface, cs: int) -> None:
        x, y = self.col * cs, self.row * cs
        pygame.draw.rect(surface, self.color, (x, y, cs, cs))
        if self.color == START_COLOR:
            cx, cy = x + cs // 2, y + cs // 2
            r = max(3, cs // 3)
            pts = [(cx - r, cy - r), (cx + r, cy), (cx - r, cy + r)]
            pygame.draw.polygon(surface, (200, 255, 200), pts)
            pygame.draw.polygon(surface, START_COLOR, pts, width=max(1, r // 4))
        elif self.color == GOAL_COLOR:
            cx, cy = x + cs // 2, y + cs // 2
            r = max(3, cs // 3)
            pygame.draw.circle(surface, (255, 200, 200), (cx, cy), r)
            pygame.draw.circle(surface, GOAL_COLOR, (cx, cy), r, width=max(1, r // 3))
            pygame.draw.circle(surface, GOAL_COLOR, (cx, cy), max(1, r // 2))


# ══════════════════════════════════════════════════════════════
# GRID
# ══════════════════════════════════════════════════════════════
class Grid:
    def __init__(self, rows: int, cols: int,
                 area_w: int = 880, area_h: int = 780):
        self.rows = rows
        self.cols = cols
        self.cell_size = min(area_w // max(1, cols), area_h // max(1, rows))
        self.nodes: list[list[Node]] = [
            [Node(r, c, rows, cols) for c in range(cols)]
            for r in range(rows)
        ]
        self.start: Node = self.nodes[0][0]
        self.goal:  Node = self.nodes[rows - 1][cols - 1]
        self.start.make_start()
        self.goal.make_goal()

    def reset_path(self) -> None:
        for row in self.nodes:
            for n in row:
                if n.is_visited() or n.is_frontier() or n.is_path():
                    n.reset()
        self.start.make_start()
        self.goal.make_goal()

    def clear(self) -> None:
        for row in self.nodes:
            for n in row:
                n.reset(); n.g = float("inf")
                n.h = 0.0; n.f = 0.0
                n.parent = None; n.neighbors = []
        self.start.make_start()
        self.goal.make_goal()

    def generate_maze(self, density: int) -> None:
        for _ in range(50):
            self.clear()
            for r in range(self.rows):
                for c in range(self.cols):
                    nd = self.nodes[r][c]
                    if nd is self.start or nd is self.goal:
                        continue
                    if random.random() < density / 100:
                        nd.make_wall()
            if self._path_exists():
                return
        self.clear()

    def _path_exists(self) -> bool:
        visited = {self.start}
        queue = [self.start]
        while queue:
            cur = queue.pop(0)
            if cur is self.goal:
                return True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                r, c = cur.row + dr, cur.col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    nb = self.nodes[r][c]
                    if nb not in visited and not nb.is_wall():
                        visited.add(nb); queue.append(nb)
        return False

    def update_all_neighbors(self) -> None:
        for row in self.nodes:
            for n in row:
                n.update_neighbors(self.nodes)

    def draw(self, surface: pygame.Surface) -> None:
        cs = self.cell_size
        for row in self.nodes:
            for n in row:
                n.draw(surface, cs)
        tw, th = self.cols * cs, self.rows * cs
        for r in range(self.rows + 1):
            pygame.draw.line(surface, GRID_LINE_COL, (0, r * cs), (tw, r * cs))
        for c in range(self.cols + 1):
            pygame.draw.line(surface, GRID_LINE_COL, (c * cs, 0), (c * cs, th))
        self.start.draw(surface, cs)
        self.goal.draw(surface, cs)

    def get_node_at_pixel(self, x: int, y: int):
        cs = self.cell_size
        if cs <= 0:
            return None
        col, row = x // cs, y // cs
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.nodes[row][col]
        return None


# ══════════════════════════════════════════════════════════════
# HEURISTICS
# ══════════════════════════════════════════════════════════════
def heuristic_manhattan(a: Node, b: Node) -> float:
    return abs(a.row - b.row) + abs(a.col - b.col)

def heuristic_euclidean(a: Node, b: Node) -> float:
    return math.sqrt((a.row - b.row) ** 2 + (a.col - b.col) ** 2)


# ══════════════════════════════════════════════════════════════
# GBFS
# ══════════════════════════════════════════════════════════════
def gbfs(grid, start, goal, heuristic_fn, move_weight, metrics):
    counter = 0
    heap: list = []
    came_from: dict[Node, Node] = {}
    visited:  set[Node] = set()
    in_front: set[Node] = set()

    def push(nd):
        nonlocal counter
        h = heuristic_fn(nd, goal); nd.h = h; nd.f = h
        heapq.heappush(heap, (h, counter, nd)); counter += 1
        in_front.add(nd)

    push(start)
    while heap:
        _, _, cur = heapq.heappop(heap)
        in_front.discard(cur)
        if cur in visited:
            yield came_from, None; continue
        visited.add(cur)
        metrics["nodes_visited"] += 1
        if cur is goal:
            path = _reconstruct(came_from, start, goal)
            metrics["path_cost"] = (len(path) - 1) * move_weight
            yield came_from, path; return
        if cur is not start and cur is not goal:
            cur.make_visited()
        for nb in cur.neighbors:
            if nb not in visited and nb not in in_front:
                came_from[nb] = cur; push(nb)
                if nb is not goal: nb.make_frontier()
        yield came_from, None
    yield came_from, []


# ══════════════════════════════════════════════════════════════
# A*
# ══════════════════════════════════════════════════════════════
def astar(grid, start, goal, heuristic_fn, move_weight, metrics):
    counter = 0
    heap: list = []
    came_from: dict[Node, Node] = {}
    g_score:  dict[Node, float] = {start: 0.0}
    in_open:  set[Node] = set()

    def push(nd, fv):
        nonlocal counter
        heapq.heappush(heap, (fv, counter, nd)); counter += 1
        in_open.add(nd)

    h0 = heuristic_fn(start, goal); start.h = h0; start.f = h0
    push(start, h0)
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur not in g_score:
            yield came_from, None; continue
        cg = g_score[cur]
        if cur is goal:
            path = _reconstruct(came_from, start, goal)
            metrics["path_cost"] = round(g_score[goal], 4)
            metrics["nodes_visited"] = len(g_score)
            yield came_from, path; return
        if cur is not start and cur is not goal:
            cur.make_visited()
        for nb in cur.neighbors:
            tg = cg + move_weight
            if tg < g_score.get(nb, float("inf")):
                came_from[nb] = cur; g_score[nb] = tg
                hv = heuristic_fn(nb, goal); fv = tg + hv
                nb.h = hv; nb.f = fv
                metrics["nodes_visited"] += 1
                push(nb, fv)
                if nb is not goal: nb.make_frontier()
        yield came_from, None
    yield came_from, []


def _reconstruct(came_from, start, goal):
    path, nd = [], goal
    while nd in came_from:
        path.append(nd); nd = came_from[nd]
    path.append(start); path.reverse()
    return path


# ══════════════════════════════════════════════════════════════
# DYNAMIC OBSTACLES
# ══════════════════════════════════════════════════════════════
class DynamicObstacleManager:
    SPAWN_PROB = 0.03
    FLASH_MS   = 300

    def __init__(self, grid: Grid):
        self.grid  = grid
        self.flash: list[tuple] = []

    def try_spawn(self, cur_node, remaining, start, goal) -> bool:
        if random.random() > self.SPAWN_PROB:
            return False
        prot = {cur_node, start, goal}
        cands = [n for row in self.grid.nodes for n in row
                 if not n.is_wall() and not n.is_start()
                 and not n.is_goal() and n not in prot]
        if not cands:
            return False
        wall = random.choice(cands)
        fc = random.choice([FLASH_RED, ACCENT_ORANGE])
        self.flash.append((wall, time.time() * 1000, fc))
        wall.color = fc
        return wall in remaining

    def commit_walls(self):
        for nd, _, _ in self.flash:
            nd.make_wall()
        self.flash.clear()

    def update_flash(self, surf, cs):
        now = time.time() * 1000
        expired = []
        for entry in self.flash:
            nd, t0, _ = entry
            el = now - t0
            if el >= self.FLASH_MS:
                nd.make_wall(); expired.append(entry)
            else:
                fc = FLASH_RED if int(el / 80) % 2 == 0 else ACCENT_ORANGE
                pygame.draw.rect(surf, fc, (nd.col * cs, nd.row * cs, cs, cs))
        for e in expired:
            self.flash.remove(e)


# ══════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════

def _rrect(surf, color, rect, r=6, w=0):
    pygame.draw.rect(surf, color, rect, border_radius=r, width=w)


class Button:
    def __init__(self, rect, label, color=BTN_DEFAULT, active_color=BTN_ACTIVE,
                 font=None, text_color=TEXT_PRIMARY):
        self.rect         = pygame.Rect(rect)
        self.label        = label
        self.color        = color
        self.active_color = active_color
        self.font         = font or FONT_MD
        self.text_color   = text_color
        self.is_active    = False
        self.hovered      = False

    def draw(self, surf):
        c = self.active_color if self.is_active else (BTN_HOVER if self.hovered else self.color)
        _rrect(surf, c, self.rect, 6)
        _rrect(surf, BTN_BORDER, self.rect, 6, 1)
        t = self.font.render(self.label, True, self.text_color)
        surf.blit(t, t.get_rect(center=self.rect.center))

    def update_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)

    def is_clicked(self, ev):
        return (ev.type == pygame.MOUSEBUTTONDOWN
                and ev.button == 1
                and self.rect.collidepoint(ev.pos))


class TextInput:
    def __init__(self, rect, initial="", label="", max_chars=6, numeric=True):
        self.rect      = pygame.Rect(rect)
        self.text      = initial
        self.label     = label
        self.max_chars = max_chars
        self.numeric   = numeric
        self.active    = False

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(ev.pos)
        if ev.type == pygame.KEYDOWN and self.active:
            if ev.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]; return False
            if ev.key == pygame.K_RETURN:
                return True
            ch = ev.unicode
            if ch and len(self.text) < self.max_chars:
                if self.numeric:
                    if ch.isdigit() or (ch == "." and "." not in self.text):
                        self.text += ch
                else:
                    self.text += ch
        return False

    def get_float(self, d=1.0):
        try: return float(self.text)
        except ValueError: return d

    def get_int(self, d=0):
        try: return int(float(self.text))
        except ValueError: return d

    def draw(self, surf):
        bc = INP_FOCUS if self.active else INP_BORDER
        _rrect(surf, INP_BG, self.rect, 4)
        _rrect(surf, bc, self.rect, 4, 1)
        if self.label:
            surf.blit(FONT_XS.render(self.label, True, TEXT_MUTED),
                      (self.rect.x, self.rect.y - 15))
        t = FONT_MONO.render(self.text + ("|" if self.active else ""), True, TEXT_PRIMARY)
        surf.blit(t, (self.rect.x + 6, self.rect.y + 4))


class SegmentedControl:
    def __init__(self, x, y, w, h, options, default=0):
        self.options  = options
        self.selected = default
        sw = w // len(options)
        self.rects = [pygame.Rect(x + i * sw, y, sw, h) for i in range(len(options))]

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for i, r in enumerate(self.rects):
                if r.collidepoint(ev.pos):
                    self.selected = i; return True
        return False

    def update_hover(self, pos): pass

    def draw(self, surf):
        for i, r in enumerate(self.rects):
            c = BTN_ACTIVE if i == self.selected else BTN_DEFAULT
            _rrect(surf, c, r, 4)
            _rrect(surf, BTN_BORDER, r, 4, 1)
            t = FONT_SM.render(self.options[i], True, TEXT_PRIMARY)
            surf.blit(t, t.get_rect(center=r.center))

    @property
    def value(self):
        return self.options[self.selected]


# ══════════════════════════════════════════════════════════════
# PANEL
# ══════════════════════════════════════════════════════════════
class Panel:
    def __init__(self, x: int, panel_h: int):
        self.x = x
        self.h = panel_h
        self.surf = pygame.Surface((PANEL_W, max(780, panel_h)))
        self._build()

    def _build(self):
        px = 16
        cw = PANEL_W - px * 2
        y  = 16

        self.title_y = y; y += 30

        # Card 1: Algorithm
        self.c1_y = y
        iy = y + 28
        bw = (cw - 6) // 2
        self.btn_gbfs  = Button((px, iy, bw, 28), "GBFS")
        self.btn_astar = Button((px + bw + 6, iy, bw, 28), "A*")
        self.btn_astar.is_active = True
        iy += 36; self.c1_h = iy - self.c1_y + 4; y = iy + 8

        # Card 2: Heuristic
        self.c2_y = y
        iy = y + 28
        self.btn_manhattan = Button((px, iy, bw, 28), "Manhattan")
        self.btn_euclidean = Button((px + bw + 6, iy, bw, 28), "Euclidean")
        self.btn_manhattan.is_active = True
        iy += 36; self.c2_h = iy - self.c2_y + 4; y = iy + 8

        # Card 3: Parameters
        self.c3_y = y
        iy = y + 10
        iw = (cw - 8) // 2
        self.inp_weight  = TextInput((px, iy + 16, iw, 24), "1",  "Weight")
        self.inp_density = TextInput((px + iw + 8, iy + 16, iw, 24), "30", "Density %")
        iy += 50
        self.speed_lbl_y = iy
        self.speed_ctrl = SegmentedControl(px, iy + 16, cw, 26,
                                           ["Slow", "Medium", "Fast"], default=1)
        iy += 50; self.c3_h = iy - self.c3_y + 4; y = iy + 8

        # Card 4: Actions
        self.c4_y = y
        iy = y + 10
        bw2 = (cw - 6) // 2
        self.btn_run      = Button((px, iy, bw2, 30), "▶  Run",
                                   color=BTN_GREEN, active_color=BTN_GREEN_HV)
        self.btn_step     = Button((px + bw2 + 6, iy, bw2, 30), "⏭  Step")
        iy += 36
        self.btn_generate = Button((px, iy, bw2, 30), "⟳  Generate")
        self.btn_reset    = Button((px + bw2 + 6, iy, bw2, 30), "↺  Reset")
        iy += 36
        self.btn_clear    = Button((px, iy, bw2, 30), "✕  Clear")
        self.btn_dynamic  = Button((px + bw2 + 6, iy, bw2, 30), "Dynamic: Off")
        iy += 38; self.c4_h = iy - self.c4_y + 4; y = iy + 8

        # Card 5: Grid Size
        self.c5_y = y
        iy = y + 10
        gw = (cw - 8) // 3
        self.inp_rows = TextInput((px, iy + 16, gw, 24), "20", "Rows")
        self.inp_cols = TextInput((px + gw + 4, iy + 16, gw, 24), "20", "Cols")
        self.btn_resize = Button((px + 2 * (gw + 4), iy + 14, gw, 28), "Apply")
        iy += 50; self.c5_h = iy - self.c5_y + 4; y = iy + 8

        # Card 6: Metrics
        self.c6_y = y
        self.c6_h = 120
        y += self.c6_h + 8

        # Card 7: Legend
        self.c7_y = y
        self.c7_h = 90
        self.status_y = y + self.c7_h + 10

    def draw(self, surface, metrics, dynamic_on, message, state_name):
        self.surf.fill(PANEL_BG)
        px = 16
        cw = PANEL_W - px * 2

        def card(cy, ch, label=""):
            _rrect(self.surf, CARD_BG, (px - 6, cy, cw + 12, ch), 8)
            _rrect(self.surf, CARD_BORDER, (px - 6, cy, cw + 12, ch), 8, 1)
            if label:
                self.surf.blit(FONT_XS.render(label, True, TEXT_MUTED), (px, cy + 8))

        # Title
        self.surf.blit(FONT_XL.render("Pathfinding Agent", True, ACCENT_CYAN),
                       (px, self.title_y))

        card(self.c1_y, self.c1_h, "ALGORITHM")
        self.btn_gbfs.draw(self.surf); self.btn_astar.draw(self.surf)

        card(self.c2_y, self.c2_h, "HEURISTIC")
        self.btn_manhattan.draw(self.surf); self.btn_euclidean.draw(self.surf)

        card(self.c3_y, self.c3_h, "PARAMETERS")
        self.inp_weight.draw(self.surf); self.inp_density.draw(self.surf)
        self.surf.blit(FONT_XS.render("Speed", True, TEXT_MUTED), (px, self.speed_lbl_y))
        self.speed_ctrl.draw(self.surf)

        card(self.c4_y, self.c4_h)
        self.btn_dynamic.label = f"Dynamic: {'On' if dynamic_on else 'Off'}"
        self.btn_dynamic.is_active = dynamic_on
        for b in (self.btn_run, self.btn_step, self.btn_generate,
                  self.btn_reset, self.btn_clear, self.btn_dynamic):
            b.draw(self.surf)

        card(self.c5_y, self.c5_h, "GRID SIZE")
        self.inp_rows.draw(self.surf); self.inp_cols.draw(self.surf)
        self.btn_resize.draw(self.surf)

        card(self.c6_y, self.c6_h, "METRICS")
        my = self.c6_y + 26
        for lbl, key in [("Nodes Visited", "nodes_visited"),
                         ("Path Cost", "path_cost"),
                         ("Time", "time_ms")]:
            self.surf.blit(FONT_SM.render(lbl, True, TEXT_SECONDARY), (px, my))
            v = str(metrics[key]) + (" ms" if key == "time_ms" else "")
            vs = FONT_MONO.render(v, True, METRIC_VAL)
            self.surf.blit(vs, (PANEL_W - px - vs.get_width() - 6, my))
            my += 22

        badge_c = {
            "idle": TEXT_MUTED, "running": ACCENT_GREEN,
            "stepping": ACCENT_AMBER, "animating": ACCENT_ORANGE,
            "moving": ACCENT_ORANGE,
        }
        bc = badge_c.get(state_name, TEXT_MUTED)
        self.surf.blit(FONT_ICON.render(f"● {state_name.upper()}", True, bc),
                       (px, self.c6_y + self.c6_h - 20))

        # Card 7: Color Legend
        card(self.c7_y, self.c7_h, "LEGEND")
        ly = self.c7_y + 24
        legend = [
            (START_COLOR,   "Start"),
            (GOAL_COLOR,    "Goal"),
            (FRONTIER_COL,  "Frontier"),
            (VISITED_COLOR, "Visited"),
            (PATH_COLOR,    "Path"),
            (WALL_COLOR,    "Wall"),
        ]
        cols_per_row = 3
        col_w = cw // cols_per_row
        for idx, (clr, txt) in enumerate(legend):
            cx = px + (idx % cols_per_row) * col_w
            cy = ly + (idx // cols_per_row) * 20
            pygame.draw.rect(self.surf, clr, (cx, cy + 2, 10, 10), border_radius=2)
            self.surf.blit(FONT_XS.render(txt, True, TEXT_SECONDARY), (cx + 14, cy))

        if message:
            mc = ACCENT_RED if "No path" in message else ACCENT_GREEN
            self.surf.blit(FONT_SM.render(message[:60], True, mc), (px, self.status_y))

        surface.blit(self.surf, (self.x, 0))

    def handle_hover(self, abs_pos):
        rel = (abs_pos[0] - self.x, abs_pos[1])
        for b in (self.btn_gbfs, self.btn_astar,
                  self.btn_manhattan, self.btn_euclidean,
                  self.btn_run, self.btn_step,
                  self.btn_generate, self.btn_reset,
                  self.btn_clear, self.btn_dynamic,
                  self.btn_resize):
            b.update_hover(rel)

    def handle_event(self, event):
        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                               pygame.KEYDOWN, pygame.KEYUP):
            return None
        le = event
        if hasattr(event, "pos"):
            le = pygame.event.Event(
                event.type,
                {**event.__dict__, "pos": (event.pos[0] - self.x, event.pos[1])})
        for inp in (self.inp_weight, self.inp_density,
                    self.inp_rows, self.inp_cols):
            inp.handle_event(le)
        self.speed_ctrl.handle_event(le)

        if self.btn_gbfs.is_clicked(le):
            self.btn_gbfs.is_active = True; self.btn_astar.is_active = False
            return "algo_gbfs"
        if self.btn_astar.is_clicked(le):
            self.btn_gbfs.is_active = False; self.btn_astar.is_active = True
            return "algo_astar"
        if self.btn_manhattan.is_clicked(le):
            self.btn_manhattan.is_active = True; self.btn_euclidean.is_active = False
            return "heu_manhattan"
        if self.btn_euclidean.is_clicked(le):
            self.btn_manhattan.is_active = False; self.btn_euclidean.is_active = True
            return "heu_euclidean"
        if self.btn_generate.is_clicked(le): return "generate"
        if self.btn_run.is_clicked(le):      return "run"
        if self.btn_step.is_clicked(le):     return "step"
        if self.btn_reset.is_clicked(le):    return "reset"
        if self.btn_clear.is_clicked(le):    return "clear"
        if self.btn_dynamic.is_clicked(le):  return "toggle_dynamic"
        if self.btn_resize.is_clicked(le):   return "resize"
        return None


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def _fresh_metrics():
    return {"nodes_visited": 0, "path_cost": 0, "time_ms": 0}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    screen = pygame.display.set_mode((INIT_W, INIT_H), pygame.RESIZABLE)
    pygame.display.set_caption("Dynamic Pathfinding Agent")
    clock = pygame.time.Clock()

    rows, cols = 20, 20

    def _grid_area():
        return max(200, screen.get_width() - PANEL_W), screen.get_height()

    gaw, gah = _grid_area()
    grid = Grid(rows, cols, gaw, gah)
    grid.generate_maze(30)

    panel = Panel(gaw, gah)

    algo_choice  = "astar"
    heuristic_fn = heuristic_manhattan
    move_weight  = 1.0
    dynamic_on   = False
    metrics      = _fresh_metrics()
    message      = ""

    algo_gen  = None
    came_from = {}
    path: list[Node] = []

    path_anim_idx = 0
    agent_idx     = 0
    agent_node    = None
    last_step     = 0.0
    start_time    = 0.0

    dyn_mgr = DynamicObstacleManager(grid)
    state   = "idle"

    def get_heuristic():
        return heuristic_euclidean if panel.btn_euclidean.is_active else heuristic_manhattan
    def get_weight():
        return max(0.1, panel.inp_weight.get_float(1.0))
    def get_density():
        return max(10, min(60, panel.inp_density.get_int(30)))
    def get_delay():
        return SPEED_MAP[panel.speed_ctrl.value]

    def start_algo():
        nonlocal algo_gen, came_from, path, state, metrics, start_time
        nonlocal path_anim_idx, agent_idx, agent_node, heuristic_fn, move_weight, message
        grid.reset_path(); grid.update_all_neighbors()
        grid.start.make_start(); grid.goal.make_goal()
        heuristic_fn = get_heuristic(); move_weight = get_weight()
        metrics = _fresh_metrics(); message = ""
        path = []; came_from = {}
        path_anim_idx = 0; agent_idx = 0; agent_node = None
        start_time = time.time() * 1000
        fn = astar if algo_choice == "astar" else gbfs
        algo_gen = fn(grid, grid.start, grid.goal, heuristic_fn, move_weight, metrics)
        state = "running"

    def step_algo():
        nonlocal algo_gen, came_from, path, state, metrics, message, path_anim_idx, agent_idx
        if algo_gen is None: return
        try:
            cf, res = next(algo_gen); came_from = cf
            if res is not None:
                metrics["time_ms"] = round(time.time() * 1000 - start_time, 1)
                if not res:
                    message = "No path found!"; state = "idle"; algo_gen = None
                else:
                    path = res; path_anim_idx = 0; state = "animating"; algo_gen = None
        except StopIteration:
            state = "idle"; algo_gen = None

    def do_replan(from_node):
        nonlocal algo_gen, came_from, path, state, metrics, start_time
        nonlocal path_anim_idx, agent_idx, message
        grid.reset_path(); grid.update_all_neighbors()
        start_time = time.time() * 1000
        metrics = _fresh_metrics(); message = "Replanning…"
        fn = astar if algo_choice == "astar" else gbfs
        algo_gen = fn(grid, from_node, grid.goal, heuristic_fn, move_weight, metrics)
        fp, fc = [], {}
        for cf, res in algo_gen:
            fc = cf
            if res is not None: fp = res; break
        metrics["time_ms"] = round(time.time() * 1000 - start_time, 1)
        algo_gen = None
        if not fp:
            message = "No path after replan!"; state = "idle"
        else:
            came_from = fc; path = fp; agent_idx = 0; path_anim_idx = 0
            for n in path:
                if n is not grid.start and n is not grid.goal: n.make_path()
            state = "moving"; message = "Replanned!"

    # ── main loop ──
    running = True
    while running:
        now = time.time() * 1000
        gaw, gah = _grid_area()
        panel.x = gaw
        grid.cell_size = min(max(1, gaw // max(1, grid.cols)),
                             max(1, gah // max(1, grid.rows)))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False; break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False; break
            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(
                    (max(event.w, PANEL_W + 300), max(event.h, 600)),
                    pygame.RESIZABLE)

            action = panel.handle_event(event)
            if action == "algo_gbfs":      algo_choice = "gbfs"
            elif action == "algo_astar":   algo_choice = "astar"
            elif action == "heu_manhattan": heuristic_fn = heuristic_manhattan
            elif action == "heu_euclidean": heuristic_fn = heuristic_euclidean
            elif action == "generate":
                if state not in ("running", "animating", "moving"):
                    d = get_density(); grid.generate_maze(d)
                    metrics = _fresh_metrics()
                    message = f"Maze generated ({d}% density)"
                    state = "idle"; algo_gen = None
            elif action == "run":
                if state in ("idle", "stepping"): start_algo()
            elif action == "step":
                if state == "idle": start_algo(); state = "stepping"
                elif state == "stepping": step_algo()
            elif action == "reset":
                if state != "running":
                    grid.reset_path(); metrics = _fresh_metrics(); message = ""
                    state = "idle"; algo_gen = None; path = []; agent_node = None
            elif action == "clear":
                grid.clear(); metrics = _fresh_metrics(); message = ""
                state = "idle"; algo_gen = None; path = []; agent_node = None
            elif action == "toggle_dynamic":
                dynamic_on = not dynamic_on; dyn_mgr = DynamicObstacleManager(grid)
            elif action == "resize":
                if state not in ("running", "animating", "moving"):
                    nr = max(5, min(50, panel.inp_rows.get_int(rows)))
                    nc = max(5, min(50, panel.inp_cols.get_int(cols)))
                    rows, cols = nr, nc
                    grid = Grid(rows, cols, *_grid_area())
                    grid.generate_maze(get_density())
                    metrics = _fresh_metrics()
                    message = f"Grid: {rows}×{cols}"
                    state = "idle"; algo_gen = None; path = []; agent_node = None
                    dyn_mgr = DynamicObstacleManager(grid)

            if state in ("idle", "stepping"):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if mx < gaw:
                        nd = grid.get_node_at_pixel(mx, my)
                        if nd:
                            mods = pygame.key.get_mods()
                            if event.button == 1:
                                if mods & pygame.KMOD_SHIFT:
                                    if nd is not grid.start and not nd.is_wall():
                                        grid.goal.reset(); grid.goal = nd; nd.make_goal()
                                else:
                                    if nd is not grid.start and nd is not grid.goal:
                                        nd.reset() if nd.is_wall() else nd.make_wall()
                            elif event.button == 3:
                                if nd is not grid.goal and not nd.is_wall():
                                    grid.start.reset(); grid.start = nd; nd.make_start()
                if event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                    mx, my = event.pos
                    if mx < gaw:
                        nd = grid.get_node_at_pixel(mx, my)
                        if nd and not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                            if nd is not grid.start and nd is not grid.goal and not nd.is_wall():
                                nd.make_wall()

        # State machine
        if state == "running":
            if now - last_step >= get_delay():
                last_step = now; step_algo()
        elif state == "animating":
            if path_anim_idx < len(path):
                n = path[path_anim_idx]
                if n is not grid.start and n is not grid.goal: n.make_path()
                path_anim_idx += 1
            else:
                agent_idx = 0; agent_node = path[0]; state = "moving"
        elif state == "moving":
            d = max(40, get_delay() * 2)
            if now - last_step >= d:
                last_step = now
                if dynamic_on and agent_idx < len(path) - 1:
                    if dyn_mgr.try_spawn(path[agent_idx], path[agent_idx:],
                                          grid.start, grid.goal):
                        dyn_mgr.commit_walls(); do_replan(path[agent_idx]); continue
                if agent_idx < len(path) - 1:
                    agent_idx += 1; agent_node = path[agent_idx]
                else:
                    agent_node = None; state = "idle"
                    message = (f"Done — Cost: {metrics['path_cost']}  "
                               f"Visited: {metrics['nodes_visited']}  "
                               f"Time: {metrics['time_ms']}ms")

        # ── Render ──
        screen.fill(BG_DARK)
        uw = grid.cols * grid.cell_size
        uh = grid.rows * grid.cell_size
        pygame.draw.rect(screen, GRID_BG, (0, 0, uw, uh))
        grid.draw(screen)
        dyn_mgr.update_flash(screen, grid.cell_size)

        if agent_node is not None:
            cs = grid.cell_size
            cx = agent_node.col * cs + cs // 2
            cy = agent_node.row * cs + cs // 2
            r  = max(4, cs // 2 - 2)
            pygame.draw.circle(screen, ACCENT_ORANGE, (cx, cy), r)
            pygame.draw.circle(screen, (200, 80, 0), (cx, cy), r, width=2)

        panel.handle_hover(pygame.mouse.get_pos())
        panel.draw(screen, metrics, dynamic_on, message, state)
        pygame.draw.line(screen, CARD_BORDER, (gaw, 0), (gaw, screen.get_height()), 1)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

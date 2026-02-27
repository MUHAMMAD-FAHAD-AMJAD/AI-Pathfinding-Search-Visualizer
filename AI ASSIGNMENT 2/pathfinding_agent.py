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
INIT_W       = 1300
INIT_H       = 800
PANEL_W      = 420
PANEL_H      = 800

# --- Color Palette ---
WHITE        = (255, 255, 255)
BLACK        = (15,  15,  15 )   # walls — slightly off-black
DARK_GREEN   = (0,   180, 0  )   # start node — bright and visible
RED          = (220, 20,  60 )   # goal node
YELLOW       = (255, 200, 0  )   # frontier
LIGHT_BLUE   = (70,  160, 220)   # visited
BRIGHT_GREEN = (0,   220, 80 )   # final path
ORANGE       = (255, 140, 0  )   # agent
GRAY         = (140, 140, 140)
LIGHT_GRAY   = (210, 210, 210)
DARK_GRAY    = (55,  55,  55 )
PANEL_BG     = (22,  22,  35 )
PANEL_BORDER = (80,  80,  120)
BTN_COLOR    = (52,  52,  85 )
BTN_HOVER    = (75,  75,  130)
BTN_ACTIVE   = (50,  120, 220)
BTN_ON       = (40,  170, 70 )
BTN_OFF      = (52,  52,  85 )
TEXT_COLOR   = (220, 220, 220)
TITLE_COLOR  = (140, 180, 255)
METRIC_BG    = (12,  12,  22 )
METRIC_VAL   = (80,  220, 140)
FLASH_RED    = (255, 40,  40 )
GRID_LINE    = (200, 200, 200)
INPUT_BG     = (38,  38,  60 )
INPUT_BORDER = (100, 100, 160)
INPUT_ACTIVE = (80,  140, 220)

# --- Fonts ---
FONT_SM   = pygame.font.SysFont("segoeui",  14)
FONT_MD   = pygame.font.SysFont("segoeui",  16)
FONT_LG   = pygame.font.SysFont("segoeui",  18, bold=True)
FONT_XL   = pygame.font.SysFont("segoeui",  22, bold=True)
FONT_MONO = pygame.font.SysFont("consolas", 15)

# --- Speed Delays (ms between algorithm steps) ---
SPEED_MAP = {"Slow": 120, "Medium": 40, "Fast": 5}

# ============================================================
# NODE CLASS
# ============================================================
class Node:
    """Represents one cell in the grid."""

    def __init__(self, row: int, col: int, total_rows: int, total_cols: int):
        self.row        = row
        self.col        = col
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.color      = WHITE
        self.neighbors: list["Node"] = []
        self.g          = float("inf")
        self.h          = 0.0
        self.f          = 0.0
        self.parent: "Node | None" = None

    # --- Comparison for heapq ---
    def __lt__(self, other: "Node") -> bool:
        return self.f < other.f

    # --- State queries ---
    def is_wall(self)     -> bool: return self.color == BLACK
    def is_start(self)    -> bool: return self.color == DARK_GREEN
    def is_goal(self)     -> bool: return self.color == RED
    def is_visited(self)  -> bool: return self.color == LIGHT_BLUE
    def is_frontier(self) -> bool: return self.color == YELLOW
    def is_path(self)     -> bool: return self.color == BRIGHT_GREEN

    # --- State setters ---
    def make_wall(self)     -> None: self.color = BLACK
    def make_start(self)    -> None: self.color = DARK_GREEN
    def make_goal(self)     -> None: self.color = RED
    def make_visited(self)  -> None: self.color = LIGHT_BLUE
    def make_frontier(self) -> None: self.color = YELLOW
    def make_path(self)     -> None: self.color = BRIGHT_GREEN
    def reset(self)         -> None: self.color = WHITE

    # --- Neighbor discovery (4-directional) ---
    def update_neighbors(self, grid: list[list["Node"]]) -> None:
        self.neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = self.row + dr, self.col + dc
            if 0 <= r < self.total_rows and 0 <= c < self.total_cols:
                if not grid[r][c].is_wall():
                    self.neighbors.append(grid[r][c])

    # --- Drawing ---
    def draw(self, surface: pygame.Surface, cell_size: int) -> None:
        x = self.col * cell_size
        y = self.row * cell_size
        pygame.draw.rect(surface, self.color, (x, y, cell_size, cell_size))
        # Draw a filled circle marker for start and goal so they are unmissable
        if self.color == DARK_GREEN or self.color == RED:
            cx = x + cell_size // 2
            cy = y + cell_size // 2
            r  = max(3, cell_size // 2 - 3)
            inner = (180, 255, 180) if self.color == DARK_GREEN else (255, 160, 160)
            pygame.draw.circle(surface, inner,           (cx, cy), r)
            pygame.draw.circle(surface, self.color,       (cx, cy), r, width=max(2, r//3))


# ============================================================
# GRID CLASS
# ============================================================
class Grid:
    """Manages the 2-D array of Nodes and all grid-level operations."""

    def __init__(self, rows: int, cols: int,
                 area_w: int = 880, area_h: int = 800):
        self.rows      = rows
        self.cols      = cols
        self.cell_size = min(area_w // cols, area_h // rows)
        self.nodes: list[list[Node]] = [
            [Node(r, c, rows, cols) for c in range(cols)]
            for r in range(rows)
        ]
        self.start: Node = self.nodes[0][0]
        self.goal:  Node = self.nodes[rows - 1][cols - 1]
        self.start.make_start()
        self.goal.make_goal()

    # --- Path reset (keeps walls) ---
    def reset_path(self) -> None:
        for row in self.nodes:
            for node in row:
                if node.is_visited() or node.is_frontier() or node.is_path():
                    node.reset()
        self.start.make_start()
        self.goal.make_goal()

    # --- Full clear ---
    def clear(self) -> None:
        for row in self.nodes:
            for node in row:
                node.reset()
                node.g = float("inf")
                node.h = 0.0
                node.f = 0.0
                node.parent = None
                node.neighbors = []
        self.start.make_start()
        self.goal.make_goal()

    # --- Random maze generation ---
    def generate_maze(self, density: int) -> None:
        """Place walls randomly, verify path exists; retry up to 50 times."""
        for attempt in range(50):
            self.clear()
            for r in range(self.rows):
                for c in range(self.cols):
                    node = self.nodes[r][c]
                    if node is self.start or node is self.goal:
                        continue
                    if random.random() < density / 100:
                        node.make_wall()
            if self._path_exists():
                return
        # If all attempts fail, clear walls so at least the grid is valid
        self.clear()

    def _path_exists(self) -> bool:
        """BFS to check connectivity of start → goal."""
        visited = set()
        queue   = [self.start]
        visited.add(self.start)
        while queue:
            current = queue.pop(0)
            if current is self.goal:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r, c = current.row + dr, current.col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    neighbor = self.nodes[r][c]
                    if neighbor not in visited and not neighbor.is_wall():
                        visited.add(neighbor)
                        queue.append(neighbor)
        return False

    # --- Update all neighbors ---
    def update_all_neighbors(self) -> None:
        for row in self.nodes:
            for node in row:
                node.update_neighbors(self.nodes)

    # --- Draw grid ---
    def draw(self, surface: pygame.Surface) -> None:
        cs = self.cell_size
        for row in self.nodes:
            for node in row:
                node.draw(surface, cs)

        # Grid lines
        total_w = self.cols * cs
        total_h = self.rows * cs
        for r in range(self.rows + 1):
            pygame.draw.line(surface, GRID_LINE, (0, r * cs), (total_w, r * cs))
        for c in range(self.cols + 1):
            pygame.draw.line(surface, GRID_LINE, (c * cs, 0), (c * cs, total_h))

        # Guarantee start/goal are always visible on top of grid lines
        self.start.make_start()
        self.goal.make_goal()
        self.start.draw(surface, cs)
        self.goal.draw(surface, cs)

    # --- Convert pixel → node ---
    def get_node_at_pixel(self, x: int, y: int) -> "Node | None":
        cs = self.cell_size
        col = x // cs
        row = y // cs
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.nodes[row][col]
        return None


# ============================================================
# HEURISTICS
# ============================================================
def heuristic_manhattan(node: Node, goal: Node) -> float:
    return abs(node.row - goal.row) + abs(node.col - goal.col)


def heuristic_euclidean(node: Node, goal: Node) -> float:
    return math.sqrt((node.row - goal.row) ** 2 + (node.col - goal.col) ** 2)


# ============================================================
# GREEDY BEST FIRST SEARCH
# ============================================================
def gbfs(
    grid:         Grid,
    start:        Node,
    goal:         Node,
    heuristic_fn,
    move_weight:  float,
    metrics:      dict,
):
    """
    Generator – yields (came_from, path | None) at each expansion step.
    Yields (came_from, path)  when goal is found.
    Yields (came_from, [])    when open list exhausted (no path).
    Yields (came_from, None)  for intermediate steps.
    """
    counter    = 0
    open_heap: list = []
    came_from: dict[Node, Node] = {}
    visited:   set[Node]        = set()
    in_frontier: set[Node]      = set()

    def push(node: Node) -> None:
        nonlocal counter
        h_val   = heuristic_fn(node, goal)
        node.h  = h_val
        node.f  = h_val
        heapq.heappush(open_heap, (h_val, counter, node))
        counter += 1
        in_frontier.add(node)

    push(start)

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        in_frontier.discard(current)

        if current in visited:
            yield came_from, None
            continue

        visited.add(current)
        metrics["nodes_visited"] += 1

        if current is goal:
            path = _reconstruct_path(came_from, start, goal)
            metrics["path_cost"] = (len(path) - 1) * move_weight
            yield came_from, path
            return

        if current is not start and current is not goal:
            current.make_visited()

        for neighbor in current.neighbors:
            if neighbor not in visited and neighbor not in in_frontier:
                came_from[neighbor] = current
                push(neighbor)
                if neighbor is not goal:
                    neighbor.make_frontier()

        yield came_from, None

    yield came_from, []   # no path


# ============================================================
# A STAR SEARCH
# ============================================================
def astar(
    grid:         Grid,
    start:        Node,
    goal:         Node,
    heuristic_fn,
    move_weight:  float,
    metrics:      dict,
):
    """
    Generator – yields (came_from, path | None) at each expansion step.
    """
    counter   = 0
    open_heap: list = []
    came_from: dict[Node, Node] = {}
    g_score:   dict[Node, float] = {start: 0.0}
    in_open:   set[Node]         = set()

    def push(node: Node, f_val: float) -> None:
        nonlocal counter
        heapq.heappush(open_heap, (f_val, counter, node))
        counter  += 1
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

        for neighbor in current.neighbors:
            tentative_g = current_g + move_weight
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor]  = current
                g_score[neighbor]    = tentative_g
                h_val                = heuristic_fn(neighbor, goal)
                f_val                = tentative_g + h_val
                neighbor.h           = h_val
                neighbor.f           = f_val
                metrics["nodes_visited"] += 1
                push(neighbor, f_val)
                if neighbor is not goal:
                    neighbor.make_frontier()

        yield came_from, None

    yield came_from, []   # no path


# ============================================================
# PATH RECONSTRUCTION (shared utility)
# ============================================================
def _reconstruct_path(
    came_from: dict[Node, Node],
    start:     Node,
    goal:      Node,
) -> list[Node]:
    path   = []
    node   = goal
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path


# ============================================================
# DYNAMIC OBSTACLES
# ============================================================
class DynamicObstacleManager:
    """
    Periodically spawns walls while the agent is moving.
    Checks if a new wall collides with the remaining path
    and signals a replan if needed.
    """
    SPAWN_PROBABILITY = 0.03   # per-frame chance
    FLASH_DURATION_MS = 300    # ms the new wall flashes red/orange

    def __init__(self, grid: Grid):
        self.grid        = grid
        self.flash_nodes: list[tuple[Node, float, tuple]] = []
        # Each entry: (node, spawn_time_ms, flash_color)

    def try_spawn(
        self,
        current_node:    Node,
        remaining_path:  list[Node],
        start:           Node,
        goal:            Node,
    ) -> bool:
        """
        Attempt to spawn a wall. Returns True if replanning is needed.
        """
        if random.random() > self.SPAWN_PROBABILITY:
            return False

        # Collect candidates: empty cells not on protected positions
        protected = {current_node, start, goal}
        candidates = [
            n for row in self.grid.nodes for n in row
            if not n.is_wall()
            and not n.is_start()
            and not n.is_goal()
            and n not in protected
        ]
        if not candidates:
            return False

        new_wall = random.choice(candidates)

        # Flash effect
        flash_color = random.choice([FLASH_RED, ORANGE])
        self.flash_nodes.append((new_wall, time.time() * 1000, flash_color))
        new_wall.color = flash_color   # temporary

        # Check if this wall is on the remaining path
        on_path = new_wall in remaining_path

        # After flash it will become BLACK (handled in update_flash)
        return on_path

    def commit_walls(self) -> None:
        """Turn all flash nodes into permanent walls."""
        for node, _, _ in self.flash_nodes:
            node.make_wall()
        self.flash_nodes.clear()

    def update_flash(self, surface: pygame.Surface, cs: int) -> None:
        """Draw flashing nodes and commit them when the flash expires."""
        now = time.time() * 1000
        expired = []
        for entry in self.flash_nodes:
            node, spawn_time, color = entry
            elapsed = now - spawn_time
            if elapsed >= self.FLASH_DURATION_MS:
                node.make_wall()
                expired.append(entry)
            else:
                # Alternate flash color
                flash = FLASH_RED if int(elapsed / 80) % 2 == 0 else ORANGE
                pygame.draw.rect(
                    surface, flash,
                    (node.col * cs, node.row * cs, cs, cs)
                )
        for e in expired:
            self.flash_nodes.remove(e)


# ============================================================
# UI PANEL AND BUTTONS
# ============================================================
class Button:
    """A clickable Pygame button with hover and active-state rendering."""

    def __init__(
        self,
        rect:         tuple[int, int, int, int],
        label:        str,
        color:        tuple = BTN_COLOR,
        active_color: tuple = BTN_ACTIVE,
        font:         pygame.font.Font = None,
    ):
        self.rect         = pygame.Rect(rect)
        self.label        = label
        self.color        = color
        self.active_color = active_color
        self.font         = font or FONT_MD
        self.is_active    = False
        self.hovered      = False

    def draw(self, surface: pygame.Surface) -> None:
        if self.is_active:
            col = self.active_color
        elif self.hovered:
            col = BTN_HOVER
        else:
            col = self.color

        pygame.draw.rect(surface, col, self.rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=1, border_radius=8)

        txt = self.font.render(self.label, True, TEXT_COLOR)
        tr  = txt.get_rect(center=self.rect.center)
        surface.blit(txt, tr)

    def update_hover(self, mouse_pos: tuple[int, int]) -> None:
        self.hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False


class TextInput:
    """A simple single-line text input box."""

    def __init__(
        self,
        rect:      tuple[int, int, int, int],
        initial:   str = "",
        label:     str = "",
        max_chars: int = 6,
        numeric:   bool = True,
    ):
        self.rect      = pygame.Rect(rect)
        self.text      = initial
        self.label     = label
        self.max_chars = max_chars
        self.numeric   = numeric
        self.active    = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns True when Enter is pressed (value confirmed)."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
                return False
            if event.key == pygame.K_RETURN:
                return True
            ch = event.unicode
            if ch and len(self.text) < self.max_chars:
                if self.numeric:
                    if ch.isdigit() or (ch == "." and "." not in self.text):
                        self.text += ch
                else:
                    self.text += ch
        return False

    def get_float(self, default: float = 1.0) -> float:
        try:
            return float(self.text)
        except ValueError:
            return default

    def get_int(self, default: int = 0) -> int:
        try:
            return int(float(self.text))
        except ValueError:
            return default

    def draw(self, surface: pygame.Surface) -> None:
        border_color = INPUT_ACTIVE if self.active else INPUT_BORDER
        pygame.draw.rect(surface, INPUT_BG,   self.rect, border_radius=4)
        pygame.draw.rect(surface, border_color, self.rect, width=1, border_radius=4)

        # Label above
        if self.label:
            lbl = FONT_SM.render(self.label, True, GRAY)
            surface.blit(lbl, (self.rect.x, self.rect.y - 18))

        # Value text
        txt = FONT_MONO.render(self.text + ("|" if self.active else ""), True, TEXT_COLOR)
        surface.blit(txt, (self.rect.x + 6, self.rect.y + 5))


class SliderButton:
    """Three-state slider displayed as three clickable segments."""

    def __init__(self, x: int, y: int, w: int, h: int, options: list[str], default: int = 1):
        self.options  = options
        self.selected = default
        seg_w = w // len(options)
        self.rects = [
            pygame.Rect(x + i * seg_w, y, seg_w, h)
            for i in range(len(options))
        ]

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, rect in enumerate(self.rects):
                if rect.collidepoint(event.pos):
                    self.selected = i
                    return True
        return False

    def update_hover(self, pos: tuple[int, int]) -> None:
        pass  # hover handled in draw

    def draw(self, surface: pygame.Surface, label: str = "") -> None:
        if label:
            lbl = FONT_SM.render(label, True, GRAY)
            surface.blit(lbl, (self.rects[0].x, self.rects[0].y - 18))

        mx, my = pygame.mouse.get_pos()
        for i, rect in enumerate(self.rects):
            hovered = rect.collidepoint(mx, my)
            if i == self.selected:
                col = BTN_ACTIVE
            elif hovered:
                col = BTN_HOVER
            else:
                col = BTN_COLOR
            pygame.draw.rect(surface, col, rect, border_radius=4)
            pygame.draw.rect(surface, PANEL_BORDER, rect, width=1, border_radius=4)
            txt = FONT_SM.render(self.options[i], True, TEXT_COLOR)
            surface.blit(txt, txt.get_rect(center=rect.center))

    @property
    def value(self) -> str:
        return self.options[self.selected]


class Panel:
    """The right-side control panel (400 × 700)."""

    def __init__(self, x: int):
        self.x   = x
        self.surf = pygame.Surface((PANEL_W, PANEL_H))

        px = 20   # left padding inside panel
        py = 14   # current y cursor

        # --- Title ---
        self.title_y = py;  py += 30

        # --- Algo buttons ---
        self.algo_label_y = py;  py += 22
        bw = (PANEL_W - px * 2 - 8) // 2
        self.btn_gbfs  = Button((px,          py, bw, 32), "GBFS")
        self.btn_astar = Button((px + bw + 8, py, bw, 32), "A* Search")
        self.btn_astar.is_active = True
        py += 40

        # --- Heuristic buttons ---
        self.heu_label_y = py;  py += 22
        bw2 = (PANEL_W - px * 2 - 8) // 2
        self.btn_manhattan  = Button((px,           py, bw2, 32), "Manhattan")
        self.btn_euclidean  = Button((px + bw2 + 8, py, bw2, 32), "Euclidean")
        self.btn_manhattan.is_active = True
        py += 40

        # --- Move Weight + Density inputs (side by side) ---
        iw = (PANEL_W - px * 2 - 10) // 2
        self.input_weight  = TextInput((px,           py + 20, iw, 28), "1", "Move Weight")
        self.input_density = TextInput((px + iw + 10, py + 20, iw, 28), "30", "Density %")
        py += 58

        # --- Speed slider ---
        self.speed_label_y = py;  py += 22
        self.speed_slider  = SliderButton(px, py, PANEL_W - px * 2, 30,
                                          ["Slow", "Medium", "Fast"], default=1)
        py += 42

        # --- Action buttons ---
        bw3 = (PANEL_W - px * 2 - 8) // 2
        self.btn_generate = Button((px,           py, bw3, 34), "GENERATE")
        self.btn_run      = Button((px + bw3 + 8, py, bw3, 34), "RUN",
                                   active_color=BTN_ON)
        py += 42

        self.btn_step     = Button((px,           py, bw3, 34), "STEP")
        self.btn_reset    = Button((px + bw3 + 8, py, bw3, 34), "RESET")
        py += 42

        self.btn_clear    = Button((px, py, bw3, 34), "CLEAR")
        self.btn_dynamic  = Button((px + bw3 + 8, py, bw3, 34), "DYNAMIC: OFF",
                                   active_color=BTN_ON)
        py += 50

        # --- Separator ---
        self.sep_y = py;  py += 20

        # --- Grid size input ---
        self.grid_label_y = py;  py += 20
        gw = (PANEL_W - px * 2 - 10) // 2
        self.input_rows = TextInput((px,          py, gw, 28), "20", "Rows")
        self.input_cols = TextInput((px + gw + 10, py, gw, 28), "20", "Cols")
        py += 52

        self.btn_resize = Button((px, py, PANEL_W - px * 2, 30), "APPLY GRID SIZE")
        py += 44

        # --- Separator ---
        self.sep2_y = py;  py += 18

        # --- Metrics ---
        self.metrics_y = py

    def draw(
        self,
        surface:    pygame.Surface,
        metrics:    dict,
        dynamic_on: bool,
        message:    str,
    ) -> None:
        self.surf.fill(PANEL_BG)
        px = 15
        SEP_COLOR = (60, 60, 95)

        def sep(y: int) -> None:
            pygame.draw.line(self.surf, SEP_COLOR,
                             (px, y), (PANEL_W - px, y))

        # ── Title ─────────────────────────────────────────────────
        t1 = FONT_XL.render("Dynamic Pathfinding Agent", True, TITLE_COLOR)
        t2 = FONT_SM.render("Muhammad Fahad Amjad  |  24F-0005", True, GRAY)
        self.surf.blit(t1, (px, self.title_y))
        self.surf.blit(t2, (px, self.subtitle_y))
        sep(self.sep0_y)

        # ── Algorithm ─────────────────────────────────────────────
        lbl_a = FONT_MD.render("Algorithm:", True, GRAY)
        self.surf.blit(lbl_a, (px, self.algo_label_y))
        self.btn_gbfs.draw(self.surf)
        self.btn_astar.draw(self.surf)
        sep(self.sep1_y)

        # ── Heuristic ─────────────────────────────────────────────
        lbl_h = FONT_MD.render("Heuristic:", True, GRAY)
        self.surf.blit(lbl_h, (px, self.heu_label_y))
        self.btn_manhattan.draw(self.surf)
        self.btn_euclidean.draw(self.surf)
        sep(self.sep2_y)

        # ── Inputs ────────────────────────────────────────────────
        self.input_weight.draw(self.surf)
        self.input_density.draw(self.surf)
        sep(self.sep3_y)

        # ── Speed ─────────────────────────────────────────────────
        lbl_s = FONT_MD.render("Speed:", True, GRAY)
        self.surf.blit(lbl_s, (px, self.speed_label_y))
        self.speed_slider.draw(self.surf, "")
        sep(self.sep4_y)

        # ── Action buttons ────────────────────────────────────────
        self.btn_dynamic.label     = f"DYNAMIC: {'ON' if dynamic_on else 'OFF'}"
        self.btn_dynamic.is_active = dynamic_on
        for btn in [self.btn_generate, self.btn_run,
                    self.btn_step, self.btn_reset,
                    self.btn_clear, self.btn_dynamic]:
            btn.draw(self.surf)
        sep(self.sep5_y)

        # ── Grid size ─────────────────────────────────────────────
        lbl_g = FONT_MD.render("Grid Size (rows x cols):", True, GRAY)
        self.surf.blit(lbl_g, (px, self.grid_label_y))
        self.input_rows.draw(self.surf)
        self.input_cols.draw(self.surf)
        self.btn_resize.draw(self.surf)
        sep(self.sep6_y)

        # ── Metrics Dashboard ─────────────────────────────────────
        mbox_x = px - 4
        mbox_w = PANEL_W - px * 2 + 8
        mbox_h = self.status_y - self.metrics_box_y - 4
        pygame.draw.rect(self.surf, METRIC_BG,
                         (mbox_x, self.metrics_box_y, mbox_w, mbox_h),
                         border_radius=8)
        pygame.draw.rect(self.surf, (60, 60, 100),
                         (mbox_x, self.metrics_box_y, mbox_w, mbox_h),
                         width=1, border_radius=8)

        my = self.metrics_box_y + 10
        dash_t = FONT_LG.render("Metrics Dashboard", True, TITLE_COLOR)
        self.surf.blit(dash_t, (px, my))
        my += 28

        for label, key, unit in [
            ("Nodes Visited", "nodes_visited", ""),
            ("Path Cost",     "path_cost",     ""),
            ("Time (ms)",     "time_ms",       "ms"),
        ]:
            row_lbl = FONT_MD.render(f"{label}:", True, GRAY)
            row_val = FONT_MONO.render(
                f"{metrics[key]}{unit}" if unit else str(metrics[key]),
                True, METRIC_VAL
            )
            self.surf.blit(row_lbl, (px, my))
            self.surf.blit(row_val, (PANEL_W - px - row_val.get_width() - 4, my))
            my += 26

        # ── Status message ────────────────────────────────────────
        if message:
            color = (255, 100, 100) if "No path" in message else (100, 220, 100)
            # Truncate if too long
            msg_surf = FONT_MD.render(message[:52], True, color)
            self.surf.blit(msg_surf, (px, self.status_y))

        # Blit panel onto main surface
        surface.blit(self.surf, (self.x, 0))

    def handle_hover(self, abs_mouse: tuple[int, int]) -> None:
        rel = (abs_mouse[0] - self.x, abs_mouse[1])
        for btn in [
            self.btn_gbfs, self.btn_astar,
            self.btn_manhattan, self.btn_euclidean,
            self.btn_generate, self.btn_run,
            self.btn_step, self.btn_reset,
            self.btn_clear, self.btn_dynamic,
            self.btn_resize,
        ]:
            btn.update_hover(rel)
        self.speed_slider.update_hover(rel)

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """
        Returns a string action name, or None.
        Adjusts event coordinates to be relative to the panel surface.
        """
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                          pygame.KEYDOWN, pygame.KEYUP):
            # Translate mouse events to panel-local coords
            local_event = event
            if hasattr(event, "pos"):
                lx = event.pos[0] - self.x
                ly = event.pos[1]
                # Create a fake event with adjusted position
                local_event = pygame.event.Event(
                    event.type,
                    {**event.__dict__, "pos": (lx, ly)},
                )

            # Text inputs
            for inp in [self.input_weight, self.input_density,
                        self.input_rows, self.input_cols]:
                inp.handle_event(local_event)

            # Speed slider
            self.speed_slider.handle_event(local_event)

            # Buttons
            if self.btn_gbfs.is_clicked(local_event):
                self.btn_gbfs.is_active  = True
                self.btn_astar.is_active = False
                return "algo_gbfs"
            if self.btn_astar.is_clicked(local_event):
                self.btn_gbfs.is_active  = False
                self.btn_astar.is_active = True
                return "algo_astar"
            if self.btn_manhattan.is_clicked(local_event):
                self.btn_manhattan.is_active = True
                self.btn_euclidean.is_active = False
                return "heu_manhattan"
            if self.btn_euclidean.is_clicked(local_event):
                self.btn_manhattan.is_active = False
                self.btn_euclidean.is_active = True
                return "heu_euclidean"
            if self.btn_generate.is_clicked(local_event):
                return "generate"
            if self.btn_run.is_clicked(local_event):
                return "run"
            if self.btn_step.is_clicked(local_event):
                return "step"
            if self.btn_reset.is_clicked(local_event):
                return "reset"
            if self.btn_clear.is_clicked(local_event):
                return "clear"
            if self.btn_dynamic.is_clicked(local_event):
                return "toggle_dynamic"
            if self.btn_resize.is_clicked(local_event):
                return "resize"

        return None


# ============================================================
# STARTUP DIALOG
# ============================================================
def startup_dialog(screen: pygame.Surface) -> tuple[int, int]:
    """Simple startup dialog to set grid dimensions."""
    clock  = pygame.time.Clock()
    font_t = pygame.font.SysFont("segoeui", 30, bold=True)
    font_s = pygame.font.SysFont("segoeui", 16)

    cw = screen.get_width()

    inp_rows = TextInput((cw // 2 - 100, 290, 85, 36), "20", "Rows", numeric=True)
    inp_cols = TextInput((cw // 2 + 20,  290, 85, 36), "20", "Cols", numeric=True)
    btn_start = pygame.Rect(cw // 2 - 110, 355, 220, 48)

    while True:
        screen.fill((18, 18, 30))
        cw = screen.get_width()

        t = font_t.render("Dynamic Pathfinding Agent", True, TITLE_COLOR)
        screen.blit(t, t.get_rect(center=(cw // 2, 160)))

        sub = font_s.render(
            "AI 2002 – Assignment 2  |  Muhammad Fahad Amjad  |  24F-0005",
            True, GRAY)
        screen.blit(sub, sub.get_rect(center=(cw // 2, 200)))

        inst = font_s.render(
            "Set grid dimensions then click START  (default 20x20)",
            True, LIGHT_GRAY)
        screen.blit(inst, inst.get_rect(center=(cw // 2, 250)))

        inp_rows.draw(screen)
        inp_cols.draw(screen)

        mx, my = pygame.mouse.get_pos()
        hov = btn_start.collidepoint(mx, my)
        pygame.draw.rect(screen, BTN_ACTIVE if hov else BTN_COLOR, btn_start, border_radius=10)
        pygame.draw.rect(screen, PANEL_BORDER, btn_start, width=1, border_radius=10)
        st = font_t.render("START", True, WHITE)
        screen.blit(st, st.get_rect(center=btn_start.center))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            inp_rows.handle_event(event)
            inp_cols.handle_event(event)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_start.collidepoint(event.pos):
                    r = max(5, min(50, inp_rows.get_int(20)))
                    c = max(5, min(50, inp_cols.get_int(20)))
                    return r, c

            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                r = max(5, min(50, inp_rows.get_int(20)))
                c = max(5, min(50, inp_cols.get_int(20)))
                return r, c

        clock.tick(60)

    return 20, 20


# ============================================================
# METRICS DASHBOARD (state container)
# ============================================================
def _fresh_metrics() -> dict:
    return {"nodes_visited": 0, "path_cost": 0, "time_ms": 0}


# ============================================================
# MAIN LOOP
# ============================================================
def main() -> None:
    screen = pygame.display.set_mode((INIT_W, INIT_H), pygame.RESIZABLE)
    pygame.display.set_caption(
        "Dynamic Pathfinding Agent | Muhammad Fahad Amjad | 24F-0005"
    )
    clock = pygame.time.Clock()

    # --- Startup ---
    rows, cols = startup_dialog(screen)

    # --- Compute dynamic grid area ---
    def _grid_area() -> tuple[int, int]:
        """Return (grid_area_w, grid_area_h) based on current screen size."""
        return screen.get_width() - PANEL_W, screen.get_height()

    grid_area_w, grid_area_h = _grid_area()

    # --- Initial state ---
    grid = Grid(rows, cols, grid_area_w, grid_area_h)
    grid.generate_maze(30)

    panel = Panel(grid_area_w)

    algo_choice  = "astar"      # "astar" | "gbfs"
    heuristic_fn = heuristic_manhattan
    move_weight  = 1.0
    dynamic_on   = False

    metrics = _fresh_metrics()
    message = ""

    # Algorithm generator state
    algo_gen         = None
    came_from: dict  = {}
    path: list[Node] = []

    # Animation state
    path_anim_idx       = 0
    agent_idx           = 0
    agent_node: Node | None = None

    # Timing
    last_step_time = 0
    start_time_ms  = 0

    # Dynamic obstacle manager
    dyn_mgr = DynamicObstacleManager(grid)

    # States: idle | running | stepping | animating | moving
    state = "idle"

    def get_heuristic():
        if panel.btn_euclidean.is_active:
            return heuristic_euclidean
        return heuristic_manhattan

    def get_move_weight() -> float:
        return max(0.1, panel.input_weight.get_float(1.0))

    def get_density() -> int:
        return max(10, min(60, panel.input_density.get_int(30)))

    def get_speed_delay() -> int:
        return SPEED_MAP[panel.speed_slider.value]

    def start_algorithm() -> None:
        nonlocal algo_gen, came_from, path, state
        nonlocal metrics, start_time_ms, path_anim_idx, agent_idx, agent_node
        nonlocal heuristic_fn, move_weight, message

        grid.reset_path()
        grid.update_all_neighbors()
        grid.start.make_start()  # ensure start/goal survive reset
        grid.goal.make_goal()

        heuristic_fn = get_heuristic()
        move_weight  = get_move_weight()
        metrics      = _fresh_metrics()
        message      = ""
        path         = []
        came_from    = {}
        path_anim_idx = 0
        agent_idx     = 0
        agent_node    = None

        start_time_ms = time.time() * 1000

        fn = astar if algo_choice == "astar" else gbfs
        algo_gen = fn(
            grid, grid.start, grid.goal,
            heuristic_fn, move_weight, metrics
        )
        state = "running"

    def step_algorithm() -> None:
        """Advance the generator one step; handle completion."""
        nonlocal algo_gen, came_from, path, state
        nonlocal metrics, message, path_anim_idx, agent_idx

        if algo_gen is None:
            return
        try:
            cf, result = next(algo_gen)
            came_from = cf
            if result is not None:
                # Algorithm finished
                metrics["time_ms"] = round(time.time() * 1000 - start_time_ms, 1)
                if len(result) == 0:
                    message  = "No path found!"
                    state    = "idle"
                    algo_gen = None
                else:
                    path          = result
                    path_anim_idx = 0
                    state         = "animating"
                    algo_gen      = None
        except StopIteration:
            state    = "idle"
            algo_gen = None

    def do_replan(from_node: Node) -> None:
        """Replan from the agent's current position."""
        nonlocal algo_gen, came_from, path, state
        nonlocal metrics, start_time_ms, path_anim_idx, agent_idx, message

        grid.reset_path()
        grid.update_all_neighbors()

        start_time_ms = time.time() * 1000
        metrics       = _fresh_metrics()
        message       = "Replanning…"

        fn = astar if algo_choice == "astar" else gbfs
        algo_gen = fn(
            grid, from_node, grid.goal,
            heuristic_fn, move_weight, metrics
        )
        # Run to completion (no step animation during replan for efficiency)
        final_path: list[Node] = []
        final_cf:   dict        = {}
        for cf, result in algo_gen:
            final_cf = cf
            if result is not None:
                final_path = result
                break

        metrics["time_ms"] = round(time.time() * 1000 - start_time_ms, 1)
        algo_gen = None

        if not final_path:
            message = "No path found after replan!"
            state   = "idle"
        else:
            came_from     = final_cf
            path          = final_path
            agent_idx     = 0
            path_anim_idx = 0
            # Skip path animation on replan — go straight to agent moving
            for n in path:
                if n is not grid.start and n is not grid.goal:
                    n.make_path()
            state   = "moving"
            message = "Replanned!"

    # ---- Main event/render loop ----
    running = True
    while running:
        now_ms = time.time() * 1000

        # Keep panel.x and grid cell_size synced with window size
        grid_area_w, grid_area_h = _grid_area()
        panel.x = grid_area_w
        grid.cell_size = min(
            max(1, grid_area_w // grid.cols),
            max(1, grid_area_h // grid.rows)
        )

        # ---- Events ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                break

            # Window resize
            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(
                    (max(event.w, PANEL_W + 300), max(event.h, PANEL_H)),
                    pygame.RESIZABLE
                )

            # Panel events (always active)
            action = panel.handle_event(event)

            if action == "algo_gbfs":
                algo_choice = "gbfs"
            elif action == "algo_astar":
                algo_choice = "astar"
            elif action == "heu_manhattan":
                heuristic_fn = heuristic_manhattan
            elif action == "heu_euclidean":
                heuristic_fn = heuristic_euclidean
            elif action == "generate":
                if state not in ("running", "animating", "moving"):
                    d = get_density()
                    grid.generate_maze(d)
                    metrics  = _fresh_metrics()
                    message  = f"Maze generated ({d}% walls)"
                    state    = "idle"
                    algo_gen = None
            elif action == "run":
                if state in ("idle", "stepping"):
                    start_algorithm()
            elif action == "step":
                if state == "idle":
                    start_algorithm()
                    state = "stepping"
                elif state == "stepping":
                    step_algorithm()
            elif action == "reset":
                if state not in ("running",):
                    grid.reset_path()
                    metrics    = _fresh_metrics()
                    message    = ""
                    state      = "idle"
                    algo_gen   = None
                    path       = []
                    agent_node = None
            elif action == "clear":
                grid.clear()
                metrics    = _fresh_metrics()
                message    = ""
                state      = "idle"
                algo_gen   = None
                path       = []
                agent_node = None
            elif action == "toggle_dynamic":
                dynamic_on = not dynamic_on
                dyn_mgr    = DynamicObstacleManager(grid)
            elif action == "resize":
                if state not in ("running", "animating", "moving"):
                    nr = max(5, min(50, panel.input_rows.get_int(rows)))
                    nc = max(5, min(50, panel.input_cols.get_int(cols)))
                    rows = nr
                    cols = nc
                    gaw, gah = _grid_area()
                    grid       = Grid(rows, cols, gaw, gah)
                    grid.generate_maze(get_density())
                    metrics    = _fresh_metrics()
                    message    = f"Grid resized to {rows}x{cols}"
                    state      = "idle"
                    algo_gen   = None
                    path       = []
                    agent_node = None
                    dyn_mgr    = DynamicObstacleManager(grid)

            # ---- Grid mouse interaction (only when idle/stepping) ----
            if state in ("idle", "stepping"):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my_pos = event.pos
                    if mx < grid_area_w:
                        node = grid.get_node_at_pixel(mx, my_pos)
                        if node:
                            mods = pygame.key.get_mods()
                            if event.button == 1:
                                if mods & pygame.KMOD_SHIFT:
                                    if node is not grid.start and not node.is_wall():
                                        grid.goal.reset()
                                        grid.goal = node
                                        node.make_goal()
                                else:
                                    if node is not grid.start and node is not grid.goal:
                                        if node.is_wall():
                                            node.reset()
                                        else:
                                            node.make_wall()
                            elif event.button == 3:
                                if node is not grid.goal and not node.is_wall():
                                    grid.start.reset()
                                    grid.start = node
                                    node.make_start()

                if event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                    mx, my_pos = event.pos
                    if mx < grid_area_w:
                        node = grid.get_node_at_pixel(mx, my_pos)
                        if node:
                            mods = pygame.key.get_mods()
                            if not (mods & pygame.KMOD_SHIFT):
                                if node is not grid.start and node is not grid.goal:
                                    if not node.is_wall():
                                        node.make_wall()

        # ---- State machine updates ----
        if state == "running":
            delay = get_speed_delay()
            if now_ms - last_step_time >= delay:
                last_step_time = now_ms
                step_algorithm()

        elif state == "animating":
            if path_anim_idx < len(path):
                node = path[path_anim_idx]
                if node is not grid.start and node is not grid.goal:
                    node.make_path()
                path_anim_idx += 1
            else:
                agent_idx  = 0
                agent_node = path[0]
                state      = "moving"

        elif state == "moving":
            delay = max(40, get_speed_delay() * 2)
            if now_ms - last_step_time >= delay:
                last_step_time = now_ms

                if dynamic_on and agent_idx < len(path) - 1:
                    remaining = path[agent_idx:]
                    blocked = dyn_mgr.try_spawn(
                        path[agent_idx], remaining, grid.start, grid.goal
                    )
                    if blocked:
                        dyn_mgr.commit_walls()
                        do_replan(path[agent_idx])
                        continue

                if agent_idx < len(path) - 1:
                    agent_idx  += 1
                    agent_node = path[agent_idx]
                else:
                    agent_node = None
                    state      = "idle"
                    message    = (
                        f"Done! Cost: {metrics['path_cost']}  "
                        f"Nodes: {metrics['nodes_visited']}  "
                        f"Time: {metrics['time_ms']}ms"
                    )

        # ---- Drawing ----
        win_w = screen.get_width()
        win_h = screen.get_height()
        screen.fill((30, 30, 30))

        # White grid background — exactly the grid's used area
        used_w = grid.cols * grid.cell_size
        used_h = grid.rows * grid.cell_size
        pygame.draw.rect(screen, WHITE, (0, 0, used_w, used_h))

        grid.draw(screen)

        # Flashing dynamic walls
        dyn_mgr.update_flash(screen, grid.cell_size)

        # Agent (orange dot)
        if agent_node is not None:
            cs = grid.cell_size
            cx = agent_node.col * cs + cs // 2
            cy = agent_node.row * cs + cs // 2
            r  = max(4, cs // 2 - 2)
            pygame.draw.circle(screen, ORANGE,       (cx, cy), r)
            pygame.draw.circle(screen, (200, 80, 0), (cx, cy), r, width=2)

        # Panel
        panel.handle_hover(pygame.mouse.get_pos())
        panel.draw(screen, metrics, dynamic_on, message)

        # Divider line between grid and panel
        pygame.draw.line(screen, (60, 60, 90),
                         (grid_area_w, 0), (grid_area_w, win_h), 2)

        # State indicator (top-left)
        state_labels = {
            "idle":      ("IDLE",      GRAY),
            "running":   ("RUNNING",   BRIGHT_GREEN),
            "stepping":  ("STEPPING",  YELLOW),
            "animating": ("ANIMATING", ORANGE),
            "moving":    ("MOVING",    ORANGE),
        }
        slabel, scol = state_labels.get(state, ("IDLE", GRAY))
        st_surf = FONT_LG.render(f"● {slabel}", True, scol)
        screen.blit(st_surf, (8, 6))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

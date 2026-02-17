import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, RadioButtons
from collections import deque
import time


grid1 = [
    [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 'S', 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, -1, 0, 0, -1, 0],
    [0, 0, 0, -1, 0, -1, 0, 'T', 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

grid2 = [
    ['S', 0, -1, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, -1, 0, -1, 0, -1, 0, -1, 0],
    [-1, 0, -1, 0, -1, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, -1, -1, -1, 0, -1, 0],
    [0, -1, -1, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, -1, -1, -1, -1, -1, -1, -1, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, -1, -1, -1, 0, -1, -1, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 'T'],
    [0, -1, -1, -1, -1, -1, -1, -1, 0, 0]
]

grid3 = [
    ['S', 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 'T']
]

scenarios = {
    'Scenario 1': grid1,
    'Scenario 2': grid2, 
    'Scenario 3': grid3
}


def find_position(grid, char):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == char:
                return (i, j)
    return None


def get_neighbors(pos, rows, cols):
    r, c = pos
    neighbors = []
    
    if r > 0:
        neighbors.append((r-1, c))
    if c < cols-1:
        neighbors.append((r, c+1))
    if r < rows-1:
        neighbors.append((r+1, c))
    if r < rows-1 and c < cols-1:
        neighbors.append((r+1, c+1))
    if c > 0:
        neighbors.append((r, c-1))
    if r > 0 and c > 0:
        neighbors.append((r-1, c-1))
    
    return neighbors


def bfs(grid, start, target, visualize_func, delay):
    rows = len(grid)
    cols = len(grid[0])
    
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    frontier = set([start])
    explored = set()
    
    if delay > 0:
        visualize_func(explored, frontier, [])
    
    while queue:
        current = queue.popleft()
        frontier.discard(current)
        explored.add(current)
        
        if delay > 0:
            visualize_func(explored, frontier, [])
        
        if current == target:
            path = []
            node = target
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, explored
        
        for neighbor in get_neighbors(current, rows, cols):
            nr, nc = neighbor
            if neighbor not in visited and grid[nr][nc] != -1:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
                frontier.add(neighbor)
        
        if delay > 0:
            time.sleep(delay)
    
    return None, explored


def dfs(grid, start, target, visualize_func, delay):
    rows = len(grid)
    cols = len(grid[0])
    
    stack = [start]
    visited = {start}
    parent = {start: None}
    frontier = set([start])
    explored = set()
    
    if delay > 0:
        visualize_func(explored, frontier, [])
    
    while stack:
        current = stack.pop()
        frontier.discard(current)
        explored.add(current)
        
        if delay > 0:
            visualize_func(explored, frontier, [])
        
        if current == target:
            path = []
            node = target
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, explored
        
        neighbors = get_neighbors(current, rows, cols)
        neighbors.reverse()
        
        for neighbor in neighbors:
            nr, nc = neighbor
            if neighbor not in visited and grid[nr][nc] != -1:
                visited.add(neighbor)
                parent[neighbor] = current
                stack.append(neighbor)
                frontier.add(neighbor)
        
        if delay > 0:
            time.sleep(delay)
    
    return None, explored


def ucs(grid, start, target, visualize_func, delay):
    rows = len(grid)
    cols = len(grid[0])
    
    import heapq
    
    pq = [(0, start)]
    visited = {start}
    parent = {start: None}
    cost = {start: 0}
    frontier = set([start])
    explored = set()
    
    if delay > 0:
        visualize_func(explored, frontier, [])
    
    while pq:
        curr_cost, current = heapq.heappop(pq)
        
        if current in explored:
            continue
        
        frontier.discard(current)
        explored.add(current)
        
        if delay > 0:
            visualize_func(explored, frontier, [])
        
        if current == target:
            path = []
            node = target
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, explored
        
        for neighbor in get_neighbors(current, rows, cols):
            nr, nc = neighbor
            if grid[nr][nc] != -1:
                r1, c1 = current
                r2, c2 = neighbor
                move_cost = 1.414 if (r1 != r2 and c1 != c2) else 1.0
                new_cost = curr_cost + move_cost
                
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    parent[neighbor] = current
                    heapq.heappush(pq, (new_cost, neighbor))
                    frontier.add(neighbor)
        
        if delay > 0:
            time.sleep(delay)
    
    return None, explored


def dls(grid, start, target, limit, visualize_func, delay):
    rows = len(grid)
    cols = len(grid[0])
    
    visited = set()
    explored = set()
    frontier = set()
    parent = {}
    
    def dls_recursive(node, depth):
        if depth > limit:
            return None
        
        visited.add(node)
        frontier.add(node)
        
        if delay > 0:
            visualize_func(explored, frontier, [])
        
        if node == target:
            return node
        
        frontier.discard(node)
        explored.add(node)
        
        if delay > 0:
            visualize_func(explored, frontier, [])
            time.sleep(delay)
        
        neighbors = get_neighbors(node, rows, cols)
        neighbors.reverse()
        
        for neighbor in neighbors:
            nr, nc = neighbor
            if neighbor not in visited and grid[nr][nc] != -1:
                parent[neighbor] = node
                result = dls_recursive(neighbor, depth + 1)
                if result is not None:
                    return result
        
        return None
    
    parent[start] = None
    result = dls_recursive(start, 0)
    
    if result:
        path = []
        node = target
        while node is not None:
            path.append(node)
            node = parent.get(node)
        path.reverse()
        return path, explored
    
    return None, explored


def iddfs(grid, start, target, visualize_func, delay):
    max_depth = 50
    
    for depth in range(max_depth):
        result, explored = dls(grid, start, target, depth, visualize_func, delay)
        if result is not None:
            return result, explored
    
    return None, set()


def bidirectional(grid, start, target, visualize_func, delay):
    rows = len(grid)
    cols = len(grid[0])
    
    queue_start = deque([start])
    queue_target = deque([target])
    
    visited_start = {start}
    visited_target = {target}
    
    parent_start = {start: None}
    parent_target = {target: None}
    
    frontier_start = set([start])
    frontier_target = set([target])
    explored = set()
    
    if delay > 0:
        visualize_func(explored, frontier_start | frontier_target, [])
    
    while queue_start and queue_target:
        current_start = queue_start.popleft()
        frontier_start.discard(current_start)
        explored.add(current_start)
        
        if delay > 0:
            visualize_func(explored, frontier_start | frontier_target, [])
        
        if current_start in visited_target:
            path1 = []
            node = current_start
            while node is not None:
                path1.append(node)
                node = parent_start[node]
            path1.reverse()
            
            path2 = []
            node = parent_target[current_start]
            while node is not None:
                path2.append(node)
                node = parent_target[node]
            
            return path1 + path2, explored
        
        for neighbor in get_neighbors(current_start, rows, cols):
            nr, nc = neighbor
            if neighbor not in visited_start and grid[nr][nc] != -1:
                visited_start.add(neighbor)
                parent_start[neighbor] = current_start
                queue_start.append(neighbor)
                frontier_start.add(neighbor)
        
        if delay > 0:
            time.sleep(delay)
        
        if not queue_target:
            break
        
        current_target = queue_target.popleft()
        frontier_target.discard(current_target)
        explored.add(current_target)
        
        if delay > 0:
            visualize_func(explored, frontier_start | frontier_target, [])
        
        if current_target in visited_start:
            path1 = []
            node = current_target
            while node is not None:
                path1.append(node)
                node = parent_start[node]
            path1.reverse()
            
            path2 = []
            node = parent_target[current_target]
            while node is not None:
                path2.append(node)
                node = parent_target[node]
            
            return path1 + path2, explored
        
        for neighbor in get_neighbors(current_target, rows, cols):
            nr, nc = neighbor
            if neighbor not in visited_target and grid[nr][nc] != -1:
                visited_target.add(neighbor)
                parent_target[neighbor] = current_target
                queue_target.append(neighbor)
                frontier_target.add(neighbor)
        
        if delay > 0:
            time.sleep(delay)
    
    return None, explored


class PathfindingVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.ax_grid = plt.subplot2grid((4, 3), (0, 0), rowspan=3, colspan=2)
        self.ax_algo = plt.subplot2grid((4, 3), (0, 2))
        self.ax_scenario = plt.subplot2grid((4, 3), (1, 2))
        self.ax_speed = plt.subplot2grid((4, 3), (2, 2))
        self.ax_button = plt.subplot2grid((4, 3), (3, 0), colspan=3)
        
        self.current_algo = 'BFS'
        self.current_scenario = 'Scenario 1'
        self.current_speed = 'Medium'
        self.is_running = False
        
        self.algo_radio = RadioButtons(self.ax_algo, 
                                       ('BFS', 'DFS', 'UCS', 'DLS', 'IDDFS', 'Bidirectional'))
        self.algo_radio.on_clicked(self.update_algo)
        
        self.scenario_radio = RadioButtons(self.ax_scenario, 
                                           ('Scenario 1', 'Scenario 2', 'Scenario 3'))
        self.scenario_radio.on_clicked(self.update_scenario)
        
        self.speed_radio = RadioButtons(self.ax_speed, 
                                        ('Slow', 'Medium', 'Fast', 'Instant'))
        self.speed_radio.on_clicked(self.update_speed)
        
        self.run_button = Button(self.ax_button, 'RUN ALGORITHM', color='#4CAF50', hovercolor='#45a049')
        self.run_button.on_clicked(self.run_algorithm)
        
        self.ax_algo.set_title('Algorithm', fontweight='bold')
        self.ax_scenario.set_title('Scenario', fontweight='bold')
        self.ax_speed.set_title('Speed', fontweight='bold')
        
        self.grid = scenarios[self.current_scenario]
        self.start = find_position(self.grid, 'S')
        self.target = find_position(self.grid, 'T')
        
        self.stats_text = None
        
        self.draw_initial_grid()
        plt.tight_layout()
    
    def update_algo(self, label):
        self.current_algo = label
    
    def update_scenario(self, label):
        self.current_scenario = label
        self.grid = scenarios[label]
        self.start = find_position(self.grid, 'S')
        self.target = find_position(self.grid, 'T')
        self.draw_initial_grid()
    
    def update_speed(self, label):
        self.current_speed = label
    
    def get_delay(self):
        speeds = {
            'Slow': 0.5,
            'Medium': 0.1,
            'Fast': 0.01,
            'Instant': 0
        }
        return speeds[self.current_speed]
    
    def draw_initial_grid(self):
        self.ax_grid.clear()
        rows = len(self.grid)
        cols = len(self.grid[0])
        
        for i in range(rows):
            for j in range(cols):
                cell = self.grid[i][j]
                if cell == -1:
                    color = '#FF6B6B'
                    self.ax_grid.add_patch(patches.Rectangle((j, rows-1-i), 1, 1, 
                                                             facecolor=color, edgecolor='black', linewidth=0.5))
                    self.ax_grid.text(j+0.5, rows-1-i+0.5, '-1', ha='center', va='center', 
                                     fontsize=10, color='white', weight='bold')
                elif cell == 'S':
                    color = '#4169E1'
                    self.ax_grid.add_patch(patches.Rectangle((j, rows-1-i), 1, 1, 
                                                             facecolor=color, edgecolor='black', linewidth=2))
                    self.ax_grid.text(j+0.5, rows-1-i+0.5, 'S', ha='center', va='center', 
                                     fontsize=14, color='white', weight='bold')
                elif cell == 'T':
                    color = '#32CD32'
                    self.ax_grid.add_patch(patches.Rectangle((j, rows-1-i), 1, 1, 
                                                             facecolor=color, edgecolor='black', linewidth=2))
                    self.ax_grid.text(j+0.5, rows-1-i+0.5, 'T', ha='center', va='center', 
                                     fontsize=14, color='white', weight='bold')
                else:
                    color = '#E8F4F8'
                    self.ax_grid.add_patch(patches.Rectangle((j, rows-1-i), 1, 1, 
                                                             facecolor=color, edgecolor='black', linewidth=0.5))
        
        self.ax_grid.set_xlim(0, cols)
        self.ax_grid.set_ylim(0, rows)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.set_xticks(range(cols+1))
        self.ax_grid.set_yticks(range(rows+1))
        self.ax_grid.grid(True, alpha=0.3)
        self.ax_grid.set_title(f'{self.current_algo} - Click RUN to Start', fontsize=14, weight='bold')
        
        if self.stats_text:
            self.stats_text.remove()
            self.stats_text = None
        
        plt.draw()
    
    def visualize_step(self, explored, frontier, path):
        self.ax_grid.clear()
        rows = len(self.grid)
        cols = len(self.grid[0])
        
        for i in range(rows):
            for j in range(cols):
                cell = self.grid[i][j]
                pos = (i, j)
                
                if pos in path and pos != self.start and pos != self.target:
                    color = '#90EE90'
                elif pos in explored and pos != self.start and pos != self.target:
                    color = '#ADD8E6'
                elif pos in frontier and pos != self.start and pos != self.target:
                    color = '#FFFF99'
                elif cell == -1:
                    color = '#FF6B6B'
                elif cell == 'S':
                    color = '#4169E1'
                elif cell == 'T':
                    color = '#32CD32'
                else:
                    color = '#E8F4F8'
                
                self.ax_grid.add_patch(patches.Rectangle((j, rows-1-i), 1, 1, 
                                                         facecolor=color, edgecolor='black', linewidth=0.5))
                
                if cell == -1:
                    self.ax_grid.text(j+0.5, rows-1-i+0.5, '-1', ha='center', va='center', 
                                     fontsize=10, color='white', weight='bold')
                elif cell == 'S':
                    self.ax_grid.text(j+0.5, rows-1-i+0.5, 'S', ha='center', va='center', 
                                     fontsize=14, color='white', weight='bold')
                elif cell == 'T':
                    self.ax_grid.text(j+0.5, rows-1-i+0.5, 'T', ha='center', va='center', 
                                     fontsize=14, color='white', weight='bold')
        
        if len(path) > 1:
            path_x = [p[1] + 0.5 for p in path]
            path_y = [rows - 1 - p[0] + 0.5 for p in path]
            self.ax_grid.plot(path_x, path_y, 'o-', color='#2E8B57', linewidth=3, 
                             markersize=8, markerfacecolor='#90EE90', markeredgecolor='#2E8B57', 
                             markeredgewidth=2)
        
        self.ax_grid.set_xlim(0, cols)
        self.ax_grid.set_ylim(0, rows)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.set_xticks(range(cols+1))
        self.ax_grid.set_yticks(range(rows+1))
        self.ax_grid.grid(True, alpha=0.3)
        
        status_text = f'{self.current_algo} - Running...'
        if len(path) > 1:
            status_text = f'{self.current_algo} - Path Found!'
        self.ax_grid.set_title(status_text, fontsize=14, weight='bold', color='green' if len(path) > 1 else 'blue')
        
        plt.pause(0.001)
    
    def run_algorithm(self, event):
        if self.is_running:
            print("Algorithm already running! Please wait...")
            return
        
        self.is_running = True
        delay = self.get_delay()
        
        algo_map = {
            'BFS': lambda: bfs(self.grid, self.start, self.target, self.visualize_step, delay),
            'DFS': lambda: dfs(self.grid, self.start, self.target, self.visualize_step, delay),
            'UCS': lambda: ucs(self.grid, self.start, self.target, self.visualize_step, delay),
            'DLS': lambda: dls(self.grid, self.start, self.target, 15, self.visualize_step, delay),
            'IDDFS': lambda: iddfs(self.grid, self.start, self.target, self.visualize_step, delay),
            'Bidirectional': lambda: bidirectional(self.grid, self.start, self.target, self.visualize_step, delay)
        }
        
        print(f"\n{'='*60}")
        print(f"Running: {self.current_algo}")
        print(f"Scenario: {self.current_scenario}")
        print(f"Speed: {self.current_speed}")
        print(f"{'='*60}")
        
        result, explored = algo_map[self.current_algo]()
        
        if result:
            print(f"\n✓ Path Found!")
            print(f"  • Nodes Explored: {len(explored)}")
            print(f"  • Path Length: {len(result)}")
            self.visualize_step(explored, set(), result)
            
            info_text = f"Algorithm: {self.current_algo}\nNodes Explored: {len(explored)}\nPath Length: {len(result)}\nStatus: ✓ Path Found!"
            self.stats_text = self.ax_grid.text(0.02, 0.98, info_text, 
                                                transform=self.ax_grid.transAxes,
                                                fontsize=11, verticalalignment='top',
                                                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
                                                fontweight='bold')
        else:
            print(f"\n✗ No Path Found!")
            print(f"  • Nodes Explored: {len(explored)}")
            self.visualize_step(explored, set(), [])
            
            info_text = f"Algorithm: {self.current_algo}\nNodes Explored: {len(explored)}\nStatus: ✗ No Path Found!"
            self.stats_text = self.ax_grid.text(0.02, 0.98, info_text, 
                                                transform=self.ax_grid.transAxes,
                                                fontsize=11, verticalalignment='top',
                                                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9),
                                                fontweight='bold')
        
        plt.draw()
        self.is_running = False
        print(f"{'='*60}\n")
    
    def show(self):
        plt.show()


def main():
    print("=" * 70)
    print(" " * 20 + "AI PATHFINDING VISUALIZER")
    print("=" * 70)
    print("\n📋 INSTRUCTIONS:")
    print("   1. Select an ALGORITHM (BFS, DFS, UCS, DLS, IDDFS, Bidirectional)")
    print("   2. Choose a SCENARIO (different grid layouts)")
    print("   3. Select SPEED (Slow/Medium/Fast/Instant)")
    print("   4. Click the GREEN 'RUN ALGORITHM' button")
    print("   5. Watch the visualization in real-time!")
    print("\n🎨 COLOR LEGEND:")
    print("   🔵 Blue (S)      = Start Point")
    print("   🟢 Green (T)     = Target Point")
    print("   🔴 Red (-1)      = Walls/Obstacles")
    print("   🟡 Yellow        = Frontier (nodes waiting to explore)")
    print("   🔵 Light Blue    = Explored (already visited)")
    print("   🟢 Green Path    = Final solution path")
    print("\n🔍 ALGORITHMS:")
    print("   • BFS: Breadth-First Search (shortest path)")
    print("   • DFS: Depth-First Search (goes deep first)")
    print("   • UCS: Uniform-Cost Search (considers move costs)")
    print("   • DLS: Depth-Limited Search (DFS with limit=15)")
    print("   • IDDFS: Iterative Deepening DFS")
    print("   • Bidirectional: Searches from both Start and Target")
    print("=" * 70)
    print("\n🚀 Starting GUI... Please wait...\n")
    
    viz = PathfindingVisualizer()
    plt.show()


if __name__ == "__main__":
    main()

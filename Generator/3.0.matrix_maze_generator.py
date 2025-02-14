import random
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque

DIRECTIONS = [(0, -2), (0, 2), (-2, 0), (2, 0)]

def ensure_odd_size(size):
    return size if size % 2 == 1 else size + 1

def create_maze(width, height, min_weight=1, max_weight=99):
    maze = np.full((height, width), 999, dtype=int)
    start_x, start_y = 1, 1
    maze[start_y, start_x] = random.randint(min_weight, max_weight)
    frontier = [(start_x + dx, start_y + dy, start_x, start_y) for dx, dy in DIRECTIONS if 1 <= start_x + dx < width - 1 and 1 <= start_y + dy < height - 1]
    
    while frontier:
        rand_index = random.randint(0, len(frontier) - 1)
        x, y, px, py = frontier.pop(rand_index)
        if maze[y, x] == 999:
            maze[y, x] = random.randint(min_weight, max_weight)
            maze[(y + py) // 2, (x + px) // 2] = random.randint(min_weight, max_weight)
            frontier.extend([(x + dx, y + dy, x, y) for dx, dy in DIRECTIONS if 1 <= x + dx < width - 1 and 1 <= y + dy < height - 1 and maze[y + dy, x + dx] == 999])
    
    return maze

def count_paths(maze, start, end):
    queue = deque([(start[0], start[1], set())])
    paths = 0
    while queue:
        x, y, visited = queue.popleft()
        if (x, y) == end:
            paths += 1
            if paths >= 5:
                return paths
        visited.add((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[1] and 0 <= ny < maze.shape[0] and maze[ny, nx] != 999 and (nx, ny) not in visited:
                queue.append((nx, ny, visited.copy()))
    return paths

def ensure_multiple_paths(maze, start, end):
    height, width = maze.shape
    walls = [(y, x) for y in range(height) for x in range(width) if maze[y, x] == 999]
    random.shuffle(walls)
    while count_paths(maze, start, end) < 5 and walls:
        y, x = walls.pop()
        maze[y, x] = random.randint(1, 99)

def remove_extra_walls(maze, ratio=0.5):
    height, width = maze.shape
    num_walls = max(1, int(min(height, width) * ratio))
    walls = [(y, x) for y in range(height) for x in range(width) if maze[y, x] == 999]
    random.shuffle(walls)
    for _ in range(min(num_walls, len(walls))):
        y, x = walls.pop()
        maze[y, x] = random.randint(1, 99)

def save_maze_as_image(maze, filename):
    parent_folder = os.path.dirname(os.getcwd())  
    folder = os.path.join(parent_folder, "Source_PIC")
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    
    n, m = maze.shape
    cmap = plt.get_cmap('YlGn')
    custom_cmap = np.array([[1, 0, 0, 1] if val == 999 else cmap(val / 100) for val in maze.flatten()])
    custom_cmap = custom_cmap.reshape((n, m, 4))
    
    fig, ax = plt.subplots(figsize=(m / 4, n / 4))
    ax.imshow(custom_cmap, interpolation='nearest')
    
    for i in range(n):
        for j in range(m):
            color = 'white' if maze[i, j] == 999 else 'black'
            ax.text(j, i, str(maze[i, j]), ha='center', va='center', fontsize=6, color=color)
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    size = 53
    size = ensure_odd_size(size)
    maze = create_maze(size, size)
    start, end = (1, 1), (size - 2, size - 2)
    ensure_multiple_paths(maze, start, end)
    remove_extra_walls(maze, ratio=0.5)
    save_maze_as_image(maze, "matrix_maze.png")

if __name__ == "__main__":
    main()

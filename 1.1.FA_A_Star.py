import pygame
import numpy as np
import heapq
import time
from math import sqrt, atan2, degrees
import os

# Initialize Pygame
pygame.init()

# Constants
CELL_SIZE = 12
WINDOW_SIZE = None

# Colors
WHITE  = (255, 255, 255)
BLACK  = (0  , 0  , 0  )
RED    = (255, 0  , 0  )
GREEN  = (0  , 255, 0  )
BLUE   = (0  , 0  , 255)  # Color for current node
YELLOW = (255, 255  , 0)  # Color for open set

class Node:
    def __init__(self, position, g, h, direction):
        self.position = position  # (x, y)
        self.g = g  # cost from start to this node
        self.h = h  # heuristic cost to target
        self.f = g + h  # total cost
        self.direction = direction  # Direction vector from parent node

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    # Euclidean distance for free-angle movement
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def get_neighbors(position, maze):
    neighbors = []
    x, y = position
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

def calculate_angle(from_node, to_node):
    dx, dy = to_node[0] - from_node[0], to_node[1] - from_node[1]
    return degrees(atan2(dy, dx))

def focal_any_angle_a_star(maze, start, target, screen):
    open_set = []
    start_node = Node(start, 0, heuristic(start, target), None)
    heapq.heappush(open_set, start_node)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, target)}
    closed_set = set()
    path = []

    while open_set:
        current = heapq.heappop(open_set)

        # Display current node and open set
        screen.fill(BLACK)
        draw_maze(maze, path, screen, open_set, closed_set, current.position)  # Draw maze with current state
        pygame.display.flip()
        #time.sleep(0.0000001)  # Delay for visual effect

        if current.position == target:
            # Reconstruct the path
            while current.position in came_from:
                path.append(current.position)
                current = came_from[current.position]
            path.append(start)
            path.reverse()

            # Draw the final path in green after the algorithm completes
            screen.fill(BLACK)  # Clear screen
            draw_maze(maze, path, screen, open_set, closed_set, current.position)  # Draw final state with path
            pygame.display.flip()
            time.sleep(0.05)  # Wait to show final path
            return path

        closed_set.add(current.position)
        neighbors = get_neighbors(current.position, maze)

        for neighbor in neighbors:
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current.position] + heuristic(current.position, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                angle = calculate_angle(current.position, neighbor)
                came_from[neighbor] = Node(current.position, g_score[current.position], f_score[current.position], angle)
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, target)

                if not any(neighbor == node.position for node in open_set):
                    heapq.heappush(open_set, Node(neighbor, tentative_g_score, heuristic(neighbor, target), angle))

    return None  # No path found

def draw_maze(maze, path, screen, open_set, closed_set, current_position):
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if maze[x, y] == 1:
                pygame.draw.rect(screen, RED, rect)  # Obstacle (red)
            else:
                pygame.draw.rect(screen, WHITE, rect)  # Free space (white)
            pygame.draw.rect(screen, BLACK, rect, 1)  # Draw grid lines

    # Highlight the current node
    if current_position:
        x, y = current_position
        rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, GREEN, rect)  # Current node (blue)

    # Highlight the nodes in the open set
    for node in open_set:
        x, y = node.position
        rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, YELLOW, rect)  # Open set (yellow)

    # Highlight the nodes in the closed set
    for pos in closed_set:
        x, y = pos
        rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, BLACK, rect)  # Closed set (black)

    # Draw the path in green if it exists
    for pos in path:
        x, y = pos
        rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, BLUE, rect)  # Path (green)

def load_maze_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        maze = np.array([[int(cell) for cell in line.strip().split()] for line in lines])
    return maze

# Load maze from file
maze = load_maze_from_file("maze.txt")
WINDOW_SIZE = (maze.shape[1] * CELL_SIZE, maze.shape[0] * CELL_SIZE + 50)  # Additional space for text

# Starting and target positions
start = (0, 0)  # Top-left corner
target = (maze.shape[0] - 1, maze.shape[1] - 1)  # Bottom-right corner

# Set up Pygame window
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("FA-A* Pathfinding Visualization")

# Game loop
start_time = time.time()
path = focal_any_angle_a_star(maze, start, target, screen)
end_time = time.time()

# Final display with timing
elapsed_time = end_time - start_time
font = pygame.font.SysFont(None, 24)
time_text = font.render(f"Time: {elapsed_time:.2f} seconds", True, WHITE)
text_x = (WINDOW_SIZE[0] - time_text.get_width()) // 2
text_y = WINDOW_SIZE[1] - 40
screen.blit(time_text, (text_x, text_y))
pygame.display.flip()

# Save the number of cell in line
def save_path_length_to_file(path, directory):
    if path is not None:
        output_file = os.path.join(directory, "Number_of_cell_FA_A_Star.txt")
        with open(output_file, 'w') as file:
            file.write(str(len(path)) + '\n')
            
# Save the final maze as an image
output_dir = os.path.join("RESULT", "No_Weight")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "FA_A_Star.png")
pygame.image.save(screen, output_path)

# Print results
if path is None:
    print("No path found.")
else:
    print("Path found!", path)
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    save_path_length_to_file(path, output_dir)

# Final game loop to keep window open
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quit Pygame
pygame.quit()

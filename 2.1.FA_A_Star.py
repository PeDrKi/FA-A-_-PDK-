import pygame
import math
import time
import os

# Kích thước màn hình
n = 50
cell_size = 13
width, height = n * cell_size, n * cell_size + 50

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Đọc ma trận từ file
def load_matrix_from_file(filename="matrix.txt"):
    with open(filename, "r") as f:
        return [list(map(int, line.split())) for line in f]

# Heuristic (khoảng cách Euclidean)
def heuristic(node, goal):
    dx, dy = abs(node[0] - goal[0]), abs(node[1] - goal[1])
    return math.sqrt(dx ** 2 + dy ** 2)

# Kiểm tra tầm nhìn
def line_of_sight(grid, start, end):
    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
    err = dx - dy

    while (x0, y0) != (x1, y1):
        if grid[x0][y0] == 0:
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return True

# Thuật toán FA-A* với hiển thị từng bước
def fa_a_star(grid, start, goal, screen):
    size = len(grid)
    open_set = []
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    open_set.append((f_score[start], start))
    came_from = {}
    closed_set = set()

    # Hiển thị từng bước
    while open_set:
        open_set.sort()
        _, current = open_set.pop(0)

        # Hiển thị ô đang xét
        draw_grid(screen, grid, open_set, closed_set, start, goal, path=None)
        pygame.display.flip()
        #time.sleep(0.000001)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], g_score[goal]

        closed_set.add(current)
        x, y = current

        neighbors = [
            (x + dx, y + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            if 0 <= x + dx < size and 0 <= y + dy < size
        ]

        for neighbor in neighbors:
            if line_of_sight(grid, current, neighbor) and neighbor not in closed_set:
                nx, ny = neighbor
                tentative_g_score = g_score[current] + grid[nx][ny]

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.append((f_score[neighbor], neighbor))
                    came_from[neighbor] = current

    return None, None

# Vẽ lưới
def draw_grid(screen, grid, open_set=None, closed_set=None, start=None, goal=None, path=None, total_weight=None):
    screen.fill(WHITE)
    font = pygame.font.Font(None, 14)

    for i in range(n):
        for j in range(n):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, BLACK, rect, 1)

    # Tô màu các ô đã đóng
    if closed_set:
        for (x, y) in closed_set:
            rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, GRAY, rect)

    # Tô màu các ô đang mở
    if open_set:
        for _, (x, y) in open_set:
            rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, YELLOW, rect)

    # Tô màu đường đi nếu có
    if path:
        for (x, y) in path:
            rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, BLUE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

    if start:
        rect = pygame.Rect(start[1] * cell_size, start[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, GREEN, rect)
        pygame.draw.rect(screen, BLACK, rect, 1)

    if goal:
        rect = pygame.Rect(goal[1] * cell_size, goal[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, RED, rect)
        pygame.draw.rect(screen, BLACK, rect, 1)

    for i in range(n):
        for j in range(n):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            weight = grid[i][j]
            text = font.render(str(weight), True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)

    if total_weight is not None:
        text = font.render(f"Total Weight: {total_weight}", True, BLACK)
        text_rect = text.get_rect(center=(width // 2, height - 25))
        screen.blit(text, text_rect)

# Hàm chính
def main():
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("FA-A* Pathfinding")
    running = True

    grid = load_matrix_from_file("matrix.txt")
    start, goal = (0, 0), (n - 1, n - 1)
    path, total_weight = fa_a_star(grid, start, goal, screen)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if path:
            draw_grid(screen, grid, path=path, start=start, goal=goal, total_weight=total_weight)
        else:
            draw_grid(screen, grid, start=start, goal=goal)

        pygame.display.flip()

    output_dir = os.path.join("RESULT", "Weight")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "FA_A_Star.png")
    pygame.image.save(screen, output_path)

    pygame.quit()

if __name__ == "__main__":
    main()

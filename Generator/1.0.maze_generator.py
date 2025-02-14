import random
import os
from PIL import Image, ImageDraw

def create_maze(width, height):
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    maze = [[1 for _ in range(width)] for _ in range(height)]
    start_x, start_y = random.randrange(1, width, 2), random.randrange(1, height, 2)
    maze[start_y][start_x] = 0

    walls = [(start_x + dx, start_y + dy, start_x + 2 * dx, start_y + 2 * dy)
             for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
             if 0 <= start_x + 2 * dx < width and 0 <= start_y + 2 * dy < height]
    
    while walls:
        wall_x, wall_y, nx, ny = random.choice(walls)
        walls.remove((wall_x, wall_y, nx, ny))
        if maze[ny][nx] == 1:
            maze[wall_y][wall_x] = 0
            maze[ny][nx] = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_wall_x, next_wall_y = nx + dx, ny + dy
                if (0 <= nx + 2 * dx < width and 0 <= ny + 2 * dy < height
                        and maze[ny + dy][nx + dx] == 1):
                    walls.append((nx + dx, ny + dy, nx + 2 * dx, ny + 2 * dy))
    return maze

def save_maze_to_file(maze, filename):
    parent_folder = os.path.dirname(os.getcwd())  # Lấy thư mục cha của thư mục hiện tại
    filepath = os.path.join(parent_folder, filename)

    with open(filepath, 'w') as f:
        for row in maze[1:-1]:
            f.write(' '.join(map(str, row[1:-1])) + '\n')
            
def save_maze_as_image(maze, filename, cell_size=10):
    parent_folder = os.path.dirname(os.getcwd())  
    folder = os.path.join(parent_folder, "Source_PIC")
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    height = len(maze)
    width = len(maze[0])
    img_width = width * cell_size
    img_height = height * cell_size

    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == 1:  # Vẽ tường
                x0, y0 = x * cell_size, y * cell_size
                x1, y1 = x0 + cell_size, y0 + cell_size
                draw.rectangle([x0, y0, x1, y1], fill="black")

    img.save(filepath)

# Kích thước mê cung
#######################
width  = 50           ##
       #  #            ##
height = 50           ##
#######################

maze = create_maze(width, height)
save_maze_to_file(maze, "maze.txt")
save_maze_as_image(maze, "maze.png")                    

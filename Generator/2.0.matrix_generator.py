import random
import os
import matplotlib.pyplot as plt

def create_grid(n, min_weight=1, max_weight=101):
    """Tạo lưới ngẫu nhiên với trọng số từ min_weight đến max_weight."""
    return [[random.randint(min_weight, max_weight) for _ in range(n)] for _ in range(n)]

def save_grid_to_file(grid, filename):
    """Lưu lưới vào file văn bản trong thư mục cha."""
    parent_folder = os.path.dirname(os.getcwd())  # Lấy thư mục cha
    filepath = os.path.join(parent_folder, filename)

    with open(filepath, 'w') as f:
        for row in grid:
            f.write(' '.join(map(str, row)) + '\n')

def save_grid_as_image(grid, filename):
    """Lưu lưới dưới dạng ảnh PNG trong thư mục Source_PIC của thư mục cha."""
    parent_folder = os.path.dirname(os.getcwd())  
    folder = os.path.join(parent_folder, "Source_PIC")
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    n = len(grid)
    fig, ax = plt.subplots(figsize=(n / 2, n / 2))
    ax.imshow(grid, cmap='YlGn', interpolation='nearest')

    # Hiển thị trọng số trên mỗi ô
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(grid[i][j]), ha='center', va='center', fontsize=6, color='black')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    n = 50

    # Tạo lưới và lưu vào file
    grid = create_grid(n)
    save_grid_to_file(grid, "matrix.txt")  # Lưu vào thư mục cha
    save_grid_as_image(grid, "matrix.png")  # Lưu vào thư mục Source_PIC của thư mục cha

if __name__ == "__main__":
    main()

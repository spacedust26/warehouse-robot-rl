# utils/visualize.py
import matplotlib.pyplot as plt
import numpy as np

def plot_maze(grid, path=None):
    mapping = {'.': 0, 'X': 1, 'S': 0, 'D': 0}
    arr = np.vectorize(lambda x: mapping.get(x, 0))(grid)
    fig, ax = plt.subplots()
    ax.imshow(arr, cmap='gray_r')

    if path is not None:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, marker='o')

    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    plt.gca().invert_yaxis()
    plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# from matplotlib.cbook import get_sample_data

# def plot_maze(grid, path=None, show_numbers=False):
#     """
#     Visualize grid as a maze with textures.
    
#     grid: list of lists of str ('.', 'X', 'S', 'D')
#     path: list of (row, col) tuples for agent path
#     """
#     n = len(grid)
#     m = len(grid[0])
    
#     fig, ax = plt.subplots(figsize=(m, n))
    
#     # Load textures
#     floor_img = plt.imread(get_sample_data('grace_hopper.png'))  # example texture, replace with your own
#     wall_img = plt.imread(get_sample_data('grace_hopper.png'))  # example texture
#     start_img = plt.imread(get_sample_data('ada.png'))
#     goal_img = plt.imread(get_sample_data('coins.png'))
#     path_img = plt.imread(get_sample_data('python.png'))
    
#     for i in range(n):
#         for j in range(m):
#             ch = grid[i][j]
#             x, y = j, n-1-i
            
#             if ch == 'X':
#                 ax.imshow(wall_img, extent=(x, x+1, y, y+1))
#             elif ch == 'S':
#                 ax.imshow(start_img, extent=(x, x+1, y, y+1))
#             elif ch == 'D':
#                 ax.imshow(goal_img, extent=(x, x+1, y, y+1))
#             else:
#                 ax.imshow(floor_img, extent=(x, x+1, y, y+1))
    
#     # Draw agent path
#     if path:
#         for step, (i, j) in enumerate(path):
#             x, y = j + 0.5, n-1-i + 0.5
#             ax.plot(x, y, 'o', color='deepskyblue', markersize=15, alpha=0.6)
#             if show_numbers:
#                 ax.text(x, y, str(step), color='black', ha='center', va='center', fontsize=10)
    
#     ax.set_xlim(0, m)
#     ax.set_ylim(0, n)
#     ax.set_xticks(range(m+1))
#     ax.set_yticks(range(n+1))
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_aspect('equal')
#     ax.grid(True, color='black', linewidth=2)
#     plt.gca().invert_yaxis()
#     plt.show()

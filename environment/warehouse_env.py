# environment/warehouse_env.py
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class WarehouseEnv:
    """A simple gridworld warehouse environment.

    Grid codes:
      'S' : start
      'D' : goal / delivery
      'X' : obstacle
      '.' : free cell
    """
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

    def __init__(self, grid, max_steps=200):
        self.grid = np.array(grid)
        self.n, self.m = self.grid.shape
        pos = np.argwhere(self.grid == 'S')
        if pos.size == 0:
            raise ValueError("Grid must contain a start 'S'")
        self.start = tuple(pos[0])
        posg = np.argwhere(self.grid == 'D')
        if posg.size == 0:
            raise ValueError("Grid must contain a goal 'D'")
        self.goal = tuple(posg[0])
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.state = self.start
        self.steps = 0
        self.done = False
        return self._get_obs()

    def _in_bounds(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.m

    def _is_obstacle(self, r, c):
        return self.grid[r, c] == 'X'

    def _get_obs(self):
        # Return (row, col)
        return self.state

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        dr, dc = self.ACTIONS[action]
        r, c = self.state
        nr, nc = r + dr, c + dc

        # clip to bounds
        if not self._in_bounds(nr, nc):
            nr, nc = r, c

        reward = -0.01  # small time penalty to encourage shorter paths
        info = {}

        # Hit obstacle
        if self._is_obstacle(nr, nc):
            reward += -10
            nr, nc = r, c  # remain in place
        else:
            # moved successfully
            pass

        old_dist = np.linalg.norm(np.array([r, c]) - np.array(self.goal))
        new_dist = np.linalg.norm(np.array([nr, nc]) - np.array(self.goal))
        if new_dist < old_dist:
            reward += 0.1
        else:
            reward += -0.02

        self.state = (nr, nc)
        self.steps += 1

        if self.state == self.goal:
            reward += 10
            self.done = True

        if self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, info

    def render(self, path=None):
        # Simple console render with optional path overlay
        grid = deepcopy(self.grid).astype(object)
        r, c = self.state
        grid[r, c] = 'R'  # robot
        if path is not None:
            for (pr, pc) in path:
                if grid[pr, pc] == '.':
                    grid[pr, pc] = '*'
        print('\n'.join(''.join(row) for row in grid))

    def render_image(self, path=None):
      fig, ax = plt.subplots(figsize=(5, 5))
      ax.set_xticks(np.arange(-0.5, self.m, 1))
      ax.set_yticks(np.arange(-0.5, self.n, 1))
      ax.grid(True)

      grid_img = np.zeros((self.n, self.m, 3))

      for i in range(self.n):
          for j in range(self.m):
              if self.grid[i][j] == "X":
                  grid_img[i, j] = [0.2, 0.2, 0.2]
              elif self.grid[i][j] == "S":
                  grid_img[i, j] = [0.2, 0.6, 1.0]
              elif self.grid[i][j] == "D":
                  grid_img[i, j] = [0.2, 1.0, 0.2]
              else:
                  grid_img[i, j] = [1, 1, 1]

      if path:
          for (x, y) in path:
              grid_img[x, y] = [1.0, 0.4, 0.4]

      ax.imshow(grid_img)
      plt.axis("off")

      fig.canvas.draw()

      width, height = fig.canvas.get_width_height()

      # --- FIX FOR TkAgg BACKEND ---
      argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
      argb = argb.reshape((height, width, 4))

      # ARGB → RGB
      rgb = argb[:, :, 1:4]

      plt.close(fig)
      return rgb


    def state_to_id(self, state):
        return state[0] * self.m + state[1]

    def id_to_state(self, id_):
        return (id_ // self.m, id_ % self.m)

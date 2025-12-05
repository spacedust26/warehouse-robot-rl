# algorithms/q_learning.py
import numpy as np
import random
from environment.warehouse_env import WarehouseEnv
# from utils.visualize import plot_maze
from utils.animate import save_animation

DEFAULT_GRID = [
    list("S.......X."),   # 10
    list(".XX.X....X"),   # padded to 10
    list(".....XX..."),   # padded to 10
    list(".X......DX"),   # padded to 10 
    list(".X.XXX...."),   # padded to 10
]


def train_q_learning(grid=DEFAULT_GRID, episodes=2000, alpha=0.1, gamma=0.95, epsilon=0.2):
    env = WarehouseEnv(grid)
    n_states = env.n * env.m
    n_actions = 4
    Q = np.zeros((n_states, n_actions))

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            s_id = env.state_to_id(state)
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                action = int(np.argmax(Q[s_id]))

            next_state, reward, done, _ = env.step(action)
            ns_id = env.state_to_id(next_state)

            Q[s_id, action] = Q[s_id, action] + alpha * (reward + gamma * np.max(Q[ns_id]) - Q[s_id, action])
            state = next_state

    return env, Q

def extract_policy(env, Q):
    policy = {}
    for id_ in range(env.n * env.m):
        policy[env.id_to_state(id_)] = int(np.argmax(Q[id_]))
    return policy

def get_path_from_policy(env, policy, max_steps=200):
    state = env.reset()
    path = [state]
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = policy[state]
        next_state, _, done, _ = env.step(action)
        path.append(next_state)
        state = next_state
        steps += 1
    return path

if __name__ == '__main__':
    env, Q = train_q_learning()
    policy = extract_policy(env, Q)
    path = get_path_from_policy(env, policy)
    env.render(path=path)
    print("Path:", path)
    # plot_maze(env.grid, path=path, show_numbers=True)
    save_animation(env, path, filename="q_learning_robot1.gif")


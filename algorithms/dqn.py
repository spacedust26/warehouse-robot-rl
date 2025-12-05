# algorithms/dqn.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from environment.warehouse_env import WarehouseEnv
from utils.animate import save_animation

class DQNNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
          nn.Linear(in_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, out_dim)
      )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buf)

def featurize_state(env, state):
    # one-hot encoding of position across grid
    vec = np.zeros(env.n * env.m, dtype=np.float32)
    idx = env.state_to_id(state)
    vec[idx] = 1.0
    return vec

def train_dqn(grid, episodes=800):
    env = WarehouseEnv(grid)
    in_dim = env.n * env.m
    out_dim = 4

    policy_net = DQNNet(in_dim, out_dim)
    target_net = DQNNet(in_dim, out_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(10000)

    batch_size = 64
    gamma = 0.95
    sync_freq = 20
    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.990

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            s_vec = featurize_state(env, state)
            if random.random() < eps:
                action = random.randint(0, out_dim - 1)
            else:
                with torch.no_grad():
                    qvals = policy_net(torch.from_numpy(s_vec).unsqueeze(0))
                    action = int(torch.argmax(qvals).item())

            next_state, reward, done, _ = env.step(action)
            buffer.push(s_vec, action, reward, featurize_state(env, next_state), done)
            ep_reward += reward

            if len(buffer) >= batch_size:
                s, a, r, ns, d = buffer.sample(batch_size)
                s = torch.from_numpy(s)
                a = torch.from_numpy(a).long()
                r = torch.from_numpy(r).float()
                ns = torch.from_numpy(ns)
                d = torch.from_numpy(d.astype(np.uint8)).float()

                q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    max_next = target_net(ns).max(1)[0]
                target = r + gamma * max_next * (1 - d)

                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        eps = max(eps_min, eps * eps_decay)
        if ep % sync_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes}  eps={eps:.3f}  ep_reward={ep_reward:.2f}")

    return env, policy_net

if __name__ == '__main__':
    grid = [
      list("S.......X."),   # 10
      list(".XX.X....X"),   # 10
      list(".....XX..."),   # 10
      list(".X......DX"),   # 10
      list(".X.XXX...."),   # 10
   ]
    env, net = train_dqn(grid, episodes=1500)
    # extract greedy path
    state = env.reset()
    path = [state]
    done = False
    while not done:
        s_vec = featurize_state(env, state)
        with torch.no_grad():
            qvals = net(torch.from_numpy(s_vec).unsqueeze(0))
            action = int(torch.argmax(qvals).item())
        next_state, _, done, _ = env.step(action)
        path.append(next_state)
        state = next_state
    env.render(path=path)
    print("DQN path:", path)
    save_animation(env, path, filename="dqn_robot.gif")


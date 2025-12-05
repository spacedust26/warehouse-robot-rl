# algorithms/ppo.py
"""
PPO for WarehouseEnv (cleaned & improved).
- Proper advantage computation
- Stochastic policy (softmax) without aggressive epsilon-greedy
- Safe normalization of returns and advantages
- Gradient clipping & adjustable entropy coefficient
- Diagnostic printing per episode
- Returns env + path for animation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from environment.warehouse_env import WarehouseEnv
from utils.animate import save_animation
import random

# -------------------------
# Featurizer (full-grid)
# -------------------------
def featurize_state(env, state):
    n, m = env.n, env.m
    grid_tensor = np.zeros((4, n, m), dtype=np.float32)

    for i in range(n):
        for j in range(m):
            ch = env.grid[i, j]
            if ch == "X":
                grid_tensor[1, i, j] = 1.0
            elif ch == "D":
                grid_tensor[2, i, j] = 1.0
            else:
                grid_tensor[0, i, j] = 1.0

    rx, ry = state
    grid_tensor[3, rx, ry] = 1.0

    return grid_tensor.ravel().astype(np.float32)

# -------------------------
# Actor-Critic Model
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def act(self, state_tensor):
        logits = self.actor(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, states, actions):
        logits = self.actor(states)
        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = torch.distributions.Categorical(probs)
        action_logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states).squeeze(-1)
        return action_logprobs, values, entropy

# -------------------------
# PPO Update
# -------------------------
def update_ppo(model, optimizer, memory, gamma=0.99, eps_clip=0.12,
               epochs=6, minibatch_size=64, entropy_coef=0.02,
               max_grad_norm=0.5):
    if len(memory["states"]) == 0:
        return

    device = next(model.parameters()).device

    # convert memory to tensors
    states = torch.tensor(np.array(memory["states"], dtype=np.float32), device=device)
    actions = torch.tensor(np.array(memory["actions"], dtype=np.int64), device=device)
    old_logprobs = torch.stack(memory["logprobs"]).to(device)
    rewards_np = np.array(memory["rewards"], dtype=np.float32)
    dones_np = np.array(memory["is_terminal"], dtype=np.bool_)

    # discounted returns
    returns = []
    discounted = 0.0
    for r, done in zip(reversed(rewards_np), reversed(dones_np)):
        if done:
            discounted = 0.0
        discounted = r + gamma * discounted
        returns.insert(0, discounted)
    returns = np.array(returns, dtype=np.float32)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    # normalize returns
    if returns_t.std() > 1e-8:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    else:
        returns_t = returns_t - returns_t.mean()

    dataset_size = states.shape[0]
    minibatch_size = min(minibatch_size, dataset_size)
    indices = np.arange(dataset_size)

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, minibatch_size):
            batch_idx = indices[start:start + minibatch_size]
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_logprobs = old_logprobs[batch_idx]
            batch_returns = returns_t[batch_idx]

            new_logprobs, values, entropy = model.evaluate(batch_states, batch_actions)

            # advantages = returns - value
            advantages = batch_returns - values.detach()
            if advantages.numel() > 1 and torch.std(advantages) > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                advantages = advantages - advantages.mean()

            ratios = torch.exp(new_logprobs - batch_old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * ((batch_returns - values) ** 2).mean()
            entropy_loss = -entropy_coef * entropy.mean()
            loss = actor_loss + critic_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

# -------------------------
# Training PPO
# -------------------------
def train_ppo(grid=None, episodes=600, update_every=8,
              gamma=0.99, eps_clip=0.12, device_str="cpu",
              entropy_coef=0.04, lr=1e-4):
    if grid is None:
        grid = [
            list("S.......X."),
            list(".XX.X....X"),
            list(".....XX..."),
            list(".X......DX"),
            list(".X.XXX...."),
        ]

    env = WarehouseEnv(grid)
    state_dim = env.n * env.m * 4
    action_dim = 4

    device = torch.device(device_str)
    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    memory = {"states": [], "actions": [], "logprobs": [], "rewards": [], "is_terminal": []}
    reward_history = []
    length_history = []

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        ep_len = 0
        done = False
        traj = []
        visited = set()

        while not done:
            s_vec = featurize_state(env, state)
            s_tensor = torch.from_numpy(s_vec).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_t, logprob_t, _ = model.act(s_tensor)
            action = int(action_t.item())
            logprob = logprob_t.squeeze(0)

            next_state, reward, done, _ = env.step(action)

            memory["states"].append(s_vec)
            memory["actions"].append(action)
            memory["logprobs"].append(logprob)
            memory["rewards"].append(reward)
            memory["is_terminal"].append(done)

            traj.append(tuple(state))
            visited.add(tuple(state))

            state = next_state
            ep_reward += reward
            ep_len += 1

            if ep_len >= env.max_steps:
                break

        reward_history.append(ep_reward)
        length_history.append(ep_len)

        print(f"Episode {ep+1}/{episodes} reward={ep_reward:.2f} len={ep_len} unique_states={len(visited)} sample_traj={traj[:min(len(traj),12)]}")

        if (ep + 1) % update_every == 0:
            update_ppo(model, optimizer, memory,
                       gamma=gamma, eps_clip=eps_clip,
                       epochs=6, minibatch_size=64,
                       entropy_coef=entropy_coef, max_grad_norm=0.5)
            memory = {k: [] for k in memory}

        if (ep + 1) % 50 == 0 or ep == 0:
            recent = reward_history[-50:] if len(reward_history) >= 1 else [ep_reward]
            print(f"  >>> avg reward (last 50) = {np.mean(recent):.2f}")

    # Evaluate greedy policy for path
    state = env.reset()
    path = [state]
    done = False
    for _ in range(env.max_steps):
        s_vec = featurize_state(env, state)
        s_tensor = torch.from_numpy(s_vec).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model.actor(s_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.argmax(probs + 1e-6*torch.rand_like(probs), dim=-1).item())
        next_state, _, done, _ = env.step(action)
        path.append(next_state)
        state = next_state
        if done:
            break

    repeats = sum(1 for i in range(1, len(path)) if path[i] == path[i-1])
    print(f"Final greedy path length {len(path)} repeats (staying same cell) = {repeats}")

    env.render(path=path)
    print("PPO path:", path)

    return model, reward_history, length_history, env, path


if __name__ == "__main__":
    model, rewards, lengths, env, path = train_ppo(episodes=600, update_every=8,
                                                   device_str="cpu",
                                                   entropy_coef=0.06, lr=1e-4)
    save_animation(env, path, filename="ppo_robot.gif", fps=3)

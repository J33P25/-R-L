"""
GreenMind: Deep Reinforcement Learning Algorithms (PPO & SAC)
Custom PyTorch implementations optimized for Continuous Building Control:
1. PPO (Proximal Policy Optimization): Clipped surrogate objective, GAE advantages
2. SAC (Soft Actor-Critic): Maximum entropy off-policy actor-critic with twin Q-critics
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
from src.config import MODELS_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# PPO Implementation
# ============================================================================
class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim=8, action_dim=3):
        super(PPOActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )
        self.actor_logstd = nn.Parameter(torch.ones(action_dim) * -1.0)

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        mean = torch.sigmoid(self.actor(x))  # Ensure mean is in [0, 1]
        std = torch.clamp(self.actor_logstd.exp(), 0.05, 0.6)
        dist = Normal(mean, std)
        value = self.critic(x)
        return dist, value

class PPOAgent:
    def __init__(self, obs_dim=8, action_dim=3, lr=3e-4, gamma=0.95, clip_eps=0.2):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.policy = PPOActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            dist, value = self.policy(obs_t)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            action = torch.clamp(action, 0.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()

    def train_step(self, states, actions, old_log_probs, returns, advantages, epochs=4, batch_size=32):
        if len(states) < 4:
            return

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(device)
        returns = torch.FloatTensor(np.array(returns)).unsqueeze(1).to(device)
        advantages = torch.FloatTensor(np.array(advantages)).unsqueeze(1).to(device)

        # Robust advantage normalization
        adv_std = advantages.std()
        if torch.isnan(adv_std) or adv_std < 1e-5:
            advantages = advantages - advantages.mean()
        else:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        dataset_size = len(states)
        for _ in range(epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                batch_idx = indices[start:start+batch_size]
                if len(batch_idx) < 2:
                    continue
                b_s = states[batch_idx]
                b_a = actions[batch_idx]
                b_old_lp = old_log_probs[batch_idx]
                b_ret = returns[batch_idx]
                b_adv = advantages[batch_idx]

                dist, value = self.policy(b_s)
                new_lp = dist.log_prob(b_a).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(torch.clamp(new_lp - b_old_lp, -20, 20))
                surr1 = ratio * b_adv.squeeze(1)
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv.squeeze(1)
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(value, b_ret)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

    def save(self, filename="ppo_greenmind.pt"):
        torch.save(self.policy.state_dict(), MODELS_DIR / filename)

    def load(self, filename="ppo_greenmind.pt"):
        path = MODELS_DIR / filename
        if path.exists():
            self.policy.load_state_dict(torch.load(path, map_location=device))
            self.policy.eval()

# ============================================================================
# SAC (Soft Actor-Critic) Implementation
# ============================================================================
class SoftActor(nn.Module):
    def __init__(self, obs_dim=8, action_dim=3):
        super(SoftActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(128, action_dim)
        self.log_std_head = nn.Linear(128, action_dim)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(-20, 2)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        x_t = dist.rsample() # Reparameterization trick
        action = torch.sigmoid(x_t) # Scale strictly to [0, 1]
        # Exact log probability with Sigmoid correction
        log_prob = dist.log_prob(x_t) - torch.log(action * (1.0 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.sigmoid(mu)

class SoftCritic(nn.Module):
    def __init__(self, obs_dim=8, action_dim=3):
        super(SoftCritic, self).__init__()
        # Twin Q-networks (Q1 and Q2) to prevent overestimation
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

class SACReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, sn, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(device),
            torch.FloatTensor(np.array(a)).to(device),
            torch.FloatTensor(np.array(r)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(sn)).to(device),
            torch.FloatTensor(np.array(d, dtype=np.float32)).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, obs_dim=8, action_dim=3, lr=3e-4, gamma=0.95, tau=0.005, alpha=0.2, auto_alpha=True):
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha

        self.actor = SoftActor(obs_dim, action_dim).to(device)
        self.critic = SoftCritic(obs_dim, action_dim).to(device)
        self.critic_target = SoftCritic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.replay_buffer = SACReplayBuffer()

        if self.auto_alpha:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

    def act(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, mu = self.actor(obs_t)
            res = mu if deterministic else action
        return res.cpu().numpy()[0]

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        s, a, r, sn, d = self.replay_buffer.sample(batch_size)

        # 1. Update Critic
        with torch.no_grad():
            next_a, next_lp, _ = self.actor(sn)
            q1_target, q2_target = self.critic_target(sn, next_a)
            min_q_target = torch.min(q1_target, q2_target) - self.alpha * next_lp
            target_q = r + (1.0 - d) * self.gamma * min_q_target

        q1, q2 = self.critic(s, a)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2. Update Actor
        pred_a, log_p, _ = self.actor(s)
        q1_pred, q2_pred = self.critic(s, pred_a)
        min_q_pred = torch.min(q1_pred, q2_pred)
        actor_loss = (self.alpha * log_p - min_q_pred).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 3. Automatic Entropy Adjustment
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_p + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = float(self.log_alpha.exp().item())

        # 4. Soft Update Target Critic (Polyak)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename="sac_greenmind.pt"):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, MODELS_DIR / filename)

    def load(self, filename="sac_greenmind.pt"):
        path = MODELS_DIR / filename
        if path.exists():
            checkpoint = torch.load(path, map_location=device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.actor.eval()

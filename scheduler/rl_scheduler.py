import os
import random
import logging
from datetime import datetime, timezone
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from statistics import mean

import gymnasium as gym
from gymnasium import spaces

from jobs.job import Job
from executor.executor import execute_job  # integrate real job execution

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('job_rl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class JobEnv(gym.Env):
    """
    Sliding-window job selection environment with action masking.
    Observations: (window_size, 2) array of [n, p]
    Actions: 0..window_size-1 pick job in window, window_size = No-Op
    """

    def __init__(self, job_queue, window_size=7, k1=0.5, k2=0.5, writer=None, simulate=True):
        super(JobEnv, self).__init__()
        self.original_job_queue = job_queue  # Keep original reference
        self.job_queue = []
        self.window_size = window_size
        self.writer = writer
        self.logger = logger
        self.simulate = simulate

        self.current_idx = 0
        self.k1 = k1
        self.k2 = k2
        self.base_noop_penalty = -0.5 
        self.rstep = 1.125  

        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(window_size, 2), dtype=np.float32
        )
        self.action_space = spaces.Discrete(window_size + 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset job queue from original
        self.job_queue = list(self.original_job_queue)
        
        # Reset all jobs
        for job in self.job_queue:
            job.completed = False
            job.creation_time = datetime.now(timezone.utc)
            job.completion_time = None
        
        self.completed_jobs = []
        self.collected_metrics = []  # metrics dicts from executor for reward calc
        self.current_step = 0
        self.noop_penalty = self.base_noop_penalty
        self.current_idx = 0
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        """Return padded window of [n, p]"""
        if self.current_idx >= len(self.job_queue):
            return np.zeros((self.window_size, 2), dtype=np.float32)

        end = min(self.current_idx + self.window_size, len(self.job_queue))
        window_jobs = self.job_queue[self.current_idx:end]
        padding = self.window_size - len(window_jobs)
        obs = [[j.n, j.p] for j in window_jobs] + [[0, 0]] * padding
        return np.array(obs, dtype=np.float32)

    def _window_indices(self):
        """Returns absolute job indices covered by current window"""
        return list(range(self.current_idx, min(self.current_idx + self.window_size, len(self.job_queue))))

    def valid_action_mask(self):
        """
        Returns a boolean mask indicating valid actions.
        mask[i] == True => action i is allowed
        Last action (index window_size) is No-Op and always allowed.
        """
        mask = np.zeros(self.window_size + 1, dtype=bool)
        window_idxs = self._window_indices()
        
        for i, abs_idx in enumerate(window_idxs):
            job = self.job_queue[abs_idx]
            mask[i] = not job.completed
        
        # Padded positions are invalid
        if len(window_idxs) < self.window_size:
            for i in range(len(window_idxs), self.window_size):
                mask[i] = False
        
        # No-Op always allowed
        mask[self.window_size] = True
        return mask

    def _execute_job(self, job):
        """Execute job via executor and collect metrics."""
        metrics = execute_job(job, simulate=self.simulate)
        self.collected_metrics.append(metrics)

    def step(self, action):
        """Execute action and return next observation, reward, done, info"""
        done = False
        truncated = False
        info = {}
        self.current_step += 1

        mask = self.valid_action_mask()

        if not mask[action]:
            # Illegal action penalty
            reward = -10.0
            self.current_idx += 1
            self.noop_penalty = self.base_noop_penalty
            self.logger.warning(f"Step {self.current_step}: Invalid action {action}")
            
        elif action == self.window_size:
            # No-Op selected
            reward = self.noop_penalty
            self.noop_penalty *= 1.1  # Compound penalty
            self.current_idx += 1
            self.logger.debug(f"Step {self.current_step}: NO-OP, penalty={reward:.3f}")
            
        else:
            # Valid job selection
            abs_idx = self.current_idx + action
            job = self.job_queue[abs_idx]
            
            if job.completed:
                # Should not happen with correct masking
                reward = -10.0
                self.logger.error(f"Step {self.current_step}: Tried to execute completed job {job.job_id}")
            else:
                # Execute the job and collect metrics
                self._execute_job(job)
                
                # Mark as completed
                job.completed = True
                # completion_time is already set by executor.mark_completed()
                self.completed_jobs.append(job)
                
                # Basic positive step reward; will be complemented by episode reward
                reward = self.rstep
                self.logger.info(f"Step {self.current_step}: Executed job {job.job_id} "
                               f"(n={job.n}, p={job.p})")
            
            # Reset no-op penalty after valid action
            self.noop_penalty = self.base_noop_penalty
            self.current_idx += 1

        # Check if episode is done
        all_done = all(j.completed for j in self.job_queue)
        past_end = self.current_idx >= len(self.job_queue)
        
        if all_done or past_end:
            done = True

        obs = self._get_obs() if not done else np.zeros((self.window_size, 2), dtype=np.float32)
        
        info["completed_count"] = len(self.completed_jobs)
        info["total_jobs"] = len(self.job_queue)
        info["current_idx"] = self.current_idx
        
        return obs, reward, done, truncated, info

    def compute_episode_reward(self):
        """
        Compute episode-level reward from real metrics:
        R_ep = k1 * (N / total_completion_time) + k2 * (N / avg_wait_time)
        """
        N = len(self.completed_jobs)
        if N == 0:
            self.logger.warning("Episode ended with 0 completed jobs")
            return 0.0

        # Aggregate metrics from executor
        durations = [m.get("duration_s", 0.0) for m in self.collected_metrics]
        waits = [m.get("queue_wait_s", 0.0) for m in self.collected_metrics]
        total_completion_time = float(sum(durations)) if durations else 0.0
        avg_wait = float(mean(waits)) if waits else 0.0
        # Avoid division by zero; if zero time, give small denom
        total_completion_time = max(total_completion_time, 1e-6)
        avg_wait = max(avg_wait, 1e-6)

        # Compute episode reward
        R_ep = self.k1 * (N / max(total_completion_time, 0.1)) + \
               self.k2 * (N / max(avg_wait, 0.1))

        # Log to TensorBoard if writer available
        if self.writer:
            self.writer.add_scalar("episode/total_completion_time", total_completion_time, self.current_step)
            self.writer.add_scalar("episode/avg_wait_time", avg_wait, self.current_step)
            self.writer.add_scalar("episode/episode_reward", R_ep, self.current_step)

        self.logger.info(f"Episode reward={R_ep:.4f}, Ctot={total_completion_time:.3f}, "
                        f"AvgWait={avg_wait:.3f}, Jobs={N}/{len(self.job_queue)}")
        return R_ep


class PolicyNetwork(nn.Module):
    """Policy network that outputs action logits"""
    
    def __init__(self, window_size):
        super(PolicyNetwork, self).__init__()
        self.window_size = window_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, window_size + 1)
        )

    def forward(self, x):
        return self.net(x)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps=1e-8):
    """
    Apply softmax with masking.
    logits: (..., action_dim)
    mask: boolean tensor where True = allowed
    """
    NEG_INF = -1e9
    logits_masked = logits.clone()
    logits_masked[~mask] = NEG_INF
    probs = torch.softmax(logits_masked, dim=dim)
    probs = probs * mask.float()
    denom = probs.sum(dim=dim, keepdim=True)
    denom = denom + (denom == 0).float() * eps
    probs = probs / denom
    return probs


def train_rl(
    job_queue,
    window_size=7,
    num_episodes=1000,
    gamma=0.99,
    lr=1e-3,
    tb_logdir="runs/rl_job_scheduler",
    simulate=True
):
    """Train the RL agent"""
    writer = SummaryWriter(log_dir=tb_logdir)
    env = JobEnv(job_queue, window_size, writer=writer, simulate=simulate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    policy = PolicyNetwork(window_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reward_history = []
    loss_history = []
    ma_window = 20
    running_ma = deque(maxlen=ma_window)

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            logits = policy(obs_tensor)
            
            mask_np = env.valid_action_mask()
            mask = torch.BoolTensor(mask_np).unsqueeze(0).to(device)

            probs = masked_softmax(logits, mask, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, done, truncated, info = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(float(reward))
            obs = next_obs

        # Compute episode reward bonus
        episode_reward = env.compute_episode_reward()
        
        # Add episode reward to final step
        if len(rewards) > 0:
            rewards[-1] += episode_reward

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient update
        log_probs_tensor = torch.stack(log_probs)
        loss = -torch.sum(log_probs_tensor * returns)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        total_reward = sum(rewards)
        reward_history.append(total_reward)
        loss_history.append(float(loss.item()))
        running_ma.append(total_reward)

        # TensorBoard logging
        writer.add_scalar("train/episode_reward", total_reward, episode)
        writer.add_scalar("train/loss", float(loss.item()), episode)
        writer.add_scalar("train/avg_reward_ma", float(np.mean(running_ma)), episode)
        writer.add_scalar("train/jobs_completed", info.get("completed_count", 0), episode)

        # Console logging
        if episode % 10 == 0:
            print(f"[Episode {episode}/{num_episodes}] Reward: {total_reward:.3f} | "
                  f"Loss: {loss.item():.3f} | MA({ma_window}): {np.mean(running_ma):.3f} | "
                  f"Jobs: {info.get('completed_count',0)}/{info.get('total_jobs',len(job_queue))}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "job_policy.pth")
    torch.save(policy.state_dict(), model_path)
    writer.close()
    logger.info(f"Training complete. Model saved to {model_path}")
    
    return policy, reward_history, loss_history


def test_rl(job_queue, policy, window_size=7, verbose=True):
    """Test the trained policy"""
    env = JobEnv(job_queue, window_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    executed_jobs = []
    steps = 0

    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = policy(obs_tensor)
        
        mask_np = env.valid_action_mask()
        mask = torch.BoolTensor(mask_np).unsqueeze(0).to(device)
        probs = masked_softmax(logits, mask, dim=-1)
        action = torch.argmax(probs, dim=-1).item()

        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        if action < window_size:
            # Calculate which job was executed
            abs_idx = info.get('current_idx', 0) - 1 + action
            abs_idx = max(0, min(abs_idx, len(job_queue) - 1))
            job = job_queue[abs_idx]
            executed_jobs.append(job.job_id)
            
            if verbose:
                print(f"[Step {steps}] EXECUTE Job {job.job_id} "
                      f"(n={job.n}, p={job.p}) -> reward {reward:.3f}")
        else:
            if verbose:
                print(f"[Step {steps}] NO-OP -> penalty {reward:.3f}")

    # Add episode reward
    episode_reward = env.compute_episode_reward()
    total_reward += episode_reward

    completed = len(env.completed_jobs)
    total = len(job_queue)
    
    print("-" * 60)
    print(f"Test Summary: Total Reward: {total_reward:.3f} | Steps: {steps} | "
          f"Jobs Completed: {completed}/{total}")
    if verbose and executed_jobs:
        print(f"Executed job IDs: {executed_jobs}")
    
    return total_reward, executed_jobs


if __name__ == "__main__":
    # Create job queue
    job_queue = []
    for job_id in range(1, 101):
        n = random.randint(1, 1000)
        p = random.randint(1, 10)
        creation_time = datetime.now(timezone.utc)
        stress_command = f"stress-ng --cpu {n} --timeout {p}s"
        
        job = Job(
            job_id=job_id,
            n=n,
            p=p,
            creation_time=creation_time,
            completed=False,
            completion_time=None,
            arrival_time=creation_time,
            stress_command=stress_command
        )
        job_queue.append(job)

    # Hyperparameters
    window_size = 7
    num_episodes = 500
    gamma = 0.99
    lr = 1e-3

    # Train
    logger.info("Starting training...")
    policy, rewards, losses = train_rl(
        job_queue=job_queue,
        window_size=window_size,
        num_episodes=num_episodes,
        gamma=gamma,
        lr=lr,
        tb_logdir="runs/rl_job_scheduler_v1"
    )

    # Test
    print("\n" + "=" * 60)
    print("TESTING TRAINED POLICY")
    print("=" * 60)
    test_rl(job_queue, policy, window_size=window_size, verbose=True)
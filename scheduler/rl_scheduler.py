import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from datetime import datetime, timezone
# Assume your existing Job class is imported:
from jobs.job import Job


# ----------------------------
# RL Environment
# ----------------------------
import gym
from gym import spaces

class JobEnv(gym.Env):
    def __init__(self, job_queue, window_size=7, noop_penalty=-0.5):
        """
        RL environment for sliding-window job scheduling.

        job_queue: list of Job objects refer to job.py to see format 
        window_size: size of the observation window (typically would also want to play around with this hyperparameter)
        noop_penalty: small negative reward for No-Op
        """
        super(JobEnv, self).__init__()
        self.job_queue = job_queue
        self.window_size = window_size
        self.noop_penalty = noop_penalty
        self.current_idx = 0

        # Observation: window_size x 2 ([n, p])
        self.observation_space = spaces.Box(
            low=-1e5, high=1e5, shape=(window_size, 2), dtype=np.float32
        )

        # Action: pick 1 of K jobs or No-Op
        self.action_space = spaces.Discrete(window_size + 1)

    def reset(self):
        self.current_idx = 0
        return self._get_obs()

    def _get_obs(self):
        if self.current_idx + self.window_size > len(self.job_queue):
            window_jobs = self.job_queue[self.current_idx:]
            padding = self.window_size - len(window_jobs)
            obs = [[j.n, j.p] for j in window_jobs] + [[0,0]] * padding
        else:
            window_jobs = self.job_queue[self.current_idx:self.current_idx + self.window_size]
            obs = [[j.n, j.p] for j in window_jobs]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        done = False

        # ----------------------------
        # Handle No-Op
        # ----------------------------
        if action == self.window_size:  # No-Op
            reward = self.noop_penalty
            self.noop_penalty= self.noop_penalty*1.5  # increase penalty for consecutive no-ops
            ## make noop penalty even larger so exponential increase if we keep on nooping
            #print(f"[No-Op] at idx {self.current_idx}")
        else:
            job = self.job_queue[self.current_idx + action]
            reward = self._job_cost(job)
            #print(f"[Execute] Job {job.job_id} at idx {self.current_idx + action} with cost { -reward }")
            # call CPU stress function:
            # simulate_job(job.n, job.p)

            # Slide window forward
            self.current_idx += 1
            if self.current_idx + self.window_size > len(self.job_queue):
                done = True
            ## check done = ture logic 
            self.noop_penalty = -0.5  # reset noop penalty after a real action

        obs = self._get_obs() if not done else np.zeros((self.window_size,2),dtype=np.float32)
        return obs, reward, done, {}
    ## MOST IMPORTANT PART TO PLAY AROUND WITH ## 
    def _job_cost(self, job):
        """
        Define reward/cost function. Can include:
        - Execution time
        - Resource usage
        - Queue delay
        """
        return job.n * job.p  # simple placeholder
    ## ONE final score should be where we add some metircs like avg time (inversely proportional to reward)
    ## total completion time (proportional to reward)

# ----------------------------
# Policy Network
# ----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, window_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size*2, 128),
            nn.ReLU(),
            nn.Linear(128, window_size+1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# ----------------------------
# Training function
# ----------------------------
def train_rl(job_queue, window_size=7, num_episodes=1000, gamma=0.99, lr=1e-3):
    env = JobEnv(job_queue, window_size)
    policy = PolicyNetwork(window_size)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(num_episodes):
        obs = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            probs = policy(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, done, _ = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)
            obs = next_obs

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # Policy gradient update
        loss = -torch.sum(torch.stack(log_probs) * returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} completed.")

    # Save the model
    torch.save(policy.state_dict(), "job_policy.pth")
    print("Training complete. Model saved as job_policy.pth")
    return policy

# ----------------------------
# Testing function
# ----------------------------
def test_rl(job_queue, policy, window_size=3):
    env = JobEnv(job_queue, window_size)
    obs = env.reset()
    done = False
    score = 0
    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs = policy(obs_tensor)
        action = torch.argmax(probs).item()
        obs, reward, done, _ = env.step(action)
        score+=reward
        print(f"Selected action: {action}, Reward: {reward}, Obs: {obs}")
    print(f"Total Score: {score}")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Example job queue (replace with your actual jobs)
    job_queue = []
    for job_id in range(1, 100 + 1):
        n = random.randint(1,1000)
        p = random.randint(1,10)

        creation_time = datetime.now(timezone.utc)
        arrival_time = -1  # will be measured at cpu simulaotr 
        stress_command = "stress-ng --cpu {n} --timeout {p}s"
        job = Job(
                job_id=job_id,
                n=n,
                p=p,
                creation_time=creation_time,
                completed=False,
                completion_time=None,
                arrival_time=arrival_time,
                stress_command=stress_command
        )

        print(f"[+] Created Job {job_id}: n={n}, p={p}")
        job_queue.append(job)
    window_size = 7

    # Train RL agent
    trained_policy = train_rl(job_queue, window_size, num_episodes=500)

    # Test trained policy
    print("\n=== Test Run ===")
    test_rl(job_queue, trained_policy, window_size)

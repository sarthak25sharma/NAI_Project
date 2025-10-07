import argparse
from typing import List, Protocol, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from jobs.job import Job
from jobs.producer import JobProducer
from executor.executor import execute_ordered_jobs
from scheduler.rl_scheduler import PolicyNetwork, masked_softmax, JobEnv, train_rl, test_rl


class Policy(Protocol):
    def order(self, jobs: List[Job]) -> List[Job]:
        ...


class FIFOPolicy:
    def order(self, jobs: List[Job]) -> List[Job]:
        from datetime import datetime, timezone
        return sorted(
            jobs,
            key=lambda j: ((j.creation_time or datetime.min.replace(tzinfo=timezone.utc)), j.job_id),
        )


POLICY_REGISTRY = {
    "fifo": FIFOPolicy,
    "rl": None,  # placeholder, set after RLPolicy is defined
}


class RLPolicy:
    def __init__(self, window_size: int = 7, model_path: Optional[str] = None, train: bool = False, num_episodes: int = 200, gamma: float = 0.99, lr: float = 1e-3, simulate: bool = True):
        self.window_size = window_size
        self.model_path = model_path
        self.train = train
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.lr = lr
        self.simulate = simulate
        self._policy = None

    def _ensure_policy(self, jobs: List[Job]):
        if self.train:
            policy, _, _ = train_rl(
                job_queue=jobs,
                window_size=self.window_size,
                num_episodes=self.num_episodes,
                gamma=self.gamma,
                lr=self.lr,
                simulate=self.simulate
            )
            self._policy = policy
            return

        if self._policy is not None:
            return

        if not self.model_path:
            raise ValueError("RLPolicy requires --rl-model-path when not training")

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = PolicyNetwork(self.window_size).to(device)
        state = torch.load(self.model_path, map_location=device)
        policy.load_state_dict(state)
        policy.eval()
        self._policy = policy

    def order(self, jobs: List[Job]) -> List[Job]:
        import torch
        self._ensure_policy(jobs)

        env = JobEnv(jobs, self.window_size, simulate=self.simulate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs, info = env.reset()
        done = False
        ordered_ids: List[int] = []

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = self._policy(obs_tensor)
            mask_np = env.valid_action_mask()
            mask = torch.BoolTensor(mask_np).unsqueeze(0).to(device)
            probs = masked_softmax(logits, mask, dim=-1)
            action = torch.argmax(probs, dim=-1).item()

            # If a job is chosen, record its ID in sequence
            if action < self.window_size:
                abs_indices = list(range(env.current_idx, min(env.current_idx + self.window_size, len(jobs))))
                if action < len(abs_indices):
                    chosen_abs_idx = abs_indices[action]
                    chosen_job = jobs[chosen_abs_idx]
                    if chosen_job.job_id not in ordered_ids:
                        ordered_ids.append(chosen_job.job_id)

            obs, reward, done, truncated, info = env.step(action)

        # Build final ordered list: selected first, then remaining in original order
        id_to_job = {j.job_id: j for j in jobs}
        ordered = [id_to_job[jid] for jid in ordered_ids if jid in id_to_job]
        remaining = [j for j in jobs if j.job_id not in set(ordered_ids)]
        return ordered + remaining


POLICY_REGISTRY["rl"] = RLPolicy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orchestrator with pluggable policy")
    parser.add_argument("--policy", type=str, default="fifo", choices=list(POLICY_REGISTRY.keys()))
    parser.add_argument("--generate-jobs", type=int, default=10)
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=4)
    parser.add_argument("--p-min", type=int, default=1)
    parser.add_argument("--p-max", type=int, default=10)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    # RL-specific options
    parser.add_argument("--rl-train", action="store_true", help="Train RL policy instead of loading")
    parser.add_argument("--rl-episodes", type=int, default=200, help="Training episodes for RL policy")
    parser.add_argument("--rl-window-size", type=int, default=7, help="Window size for RL policy")
    parser.add_argument("--rl-model-path", type=str, default=None, help="Path to load a saved RL policy")
    parser.add_argument("--rl-simulate", action="store_true", help="Simulate job execution during RL training/testing")
    parser.add_argument("--rl-gamma", type=float, default=0.99)
    parser.add_argument("--rl-lr", type=float, default=1e-3)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    # 1) Generate jobs
    producer = JobProducer(
        n_range=(args.n_min, args.n_max),
        p_range=(args.p_min, args.p_max),
        max_jobs=args.generate_jobs,
    )
    producer.produce_jobs()

    # 2) Order via policy
    policy_cls = POLICY_REGISTRY[args.policy]
    if args.policy == "rl":
        policy = policy_cls(
            window_size=args.rl_window_size,
            model_path=args.rl_model_path,
            train=args.rl_train,
            num_episodes=args.rl_episodes,
            gamma=args.rl_gamma,
            lr=args.rl_lr,
            simulate=args.rl_simulate or args.simulate,
        )
    else:
        policy = policy_cls()
    jobs = list(Job.registry.values())
    ordered = policy.order(jobs)

    # 3) Execute and record metrics
    execute_ordered_jobs(ordered, output_dir=args.output_dir, simulate=args.simulate)


if __name__ == "__main__":
    main()



import argparse
from typing import List, Protocol

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from jobs.job import Job
from jobs.producer import JobProducer
from executor.executor import execute_ordered_jobs


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
}


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
    policy = policy_cls()
    jobs = list(Job.registry.values())
    ordered = policy.order(jobs)

    # 3) Execute and record metrics
    execute_ordered_jobs(ordered, output_dir=args.output_dir, simulate=args.simulate)


if __name__ == "__main__":
    main()



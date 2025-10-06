import argparse
import csv
import json
import os
import shlex
import shutil
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Ensure project root is on sys.path when running directly
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from jobs.job import Job
from jobs.producer import JobProducer


def ensure_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def isoformat(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    # Always store as UTC ISO 8601
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def is_stress_ng_available() -> bool:
    return shutil.which("stress-ng") is not None


def build_stress_ng_args(job: Job) -> List[str]:
    # Build argv rather than using shell strings for safety
    return [
        "stress-ng",
        "--cpu",
        str(job.n),
        "--timeout",
        f"{job.p}s",
        "--metrics-brief",
    ]


def simulate_execution(seconds: int) -> Dict[str, str]:
    start = time.perf_counter()
    time.sleep(max(0, seconds))
    end = time.perf_counter()
    simulated_stdout = f"simulated run for {seconds}s\n"
    return {
        "stdout": simulated_stdout,
        "stderr": "",
        "exit_code": 0,
        "duration_s": end - start,
    }


def execute_job(job: Job, simulate: bool) -> Dict[str, object]:
    import subprocess

    start_time = datetime.now(timezone.utc)
    queue_wait_s = (start_time - job.creation_time).total_seconds()

    if simulate or not is_stress_ng_available():
        run_result = simulate_execution(job.p)
        duration_s = run_result["duration_s"]
        exit_code = run_result["exit_code"]
        stdout_text = run_result["stdout"]
        stderr_text = run_result["stderr"]
    else:
        args = build_stress_ng_args(job)
        start_perf = time.perf_counter()
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
        )
        duration_s = time.perf_counter() - start_perf
        exit_code = proc.returncode
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""

    end_time = datetime.now(timezone.utc)

    job.mark_completed()

    metrics: Dict[str, object] = {
        "job_id": job.job_id,
        "n": job.n,
        "p": job.p,
        "creation_time": isoformat(job.creation_time),
        "arrival_time": isoformat(job.arrival_time) if isinstance(job.arrival_time, datetime) else job.arrival_time,
        "queue_wait_s": round(queue_wait_s, 6),
        "start_time": isoformat(start_time),
        "end_time": isoformat(end_time),
        "duration_s": round(duration_s, 6),
        "exit_code": exit_code,
        "success": exit_code == 0,
        "stdout_len": len(stdout_text),
        "stderr_len": len(stderr_text),
    }

    # Keep large blobs optional, controlled by caller if needed later
    return metrics


def write_jsonl(metrics_list: List[Dict[str, object]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as fh:
        for record in metrics_list:
            fh.write(json.dumps(record) + "\n")


def write_csv(metrics_list: List[Dict[str, object]], output_path: str) -> None:
    if not metrics_list:
        # Create an empty file to signal no data
        open(output_path, "w").close()
        return
    fieldnames = list(metrics_list[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in metrics_list:
            writer.writerow(record)


def fifo_order(jobs: List[Job]) -> List[Job]:
    # FIFO by creation_time, fallback to job_id for stability
    return sorted(
        jobs,
        key=lambda j: ((j.creation_time or datetime.min.replace(tzinfo=timezone.utc)), j.job_id),
    )


def collect_registry_jobs() -> List[Job]:
    return list(Job.registry.values())


def generate_jobs_if_requested(args: argparse.Namespace) -> None:
    if args.generate_jobs <= 0:
        return
    producer = JobProducer(
        n_range=(args.n_min, args.n_max),
        p_range=(args.p_min, args.p_max),
        max_jobs=args.generate_jobs,
    )
    producer.produce_jobs()


def execute_ordered_jobs(ordered_jobs: List[Job], output_dir: str, simulate: bool) -> None:
    ensure_directory(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = os.path.join(output_dir, f"metrics_{timestamp}.jsonl")
    csv_path = os.path.join(output_dir, f"metrics_{timestamp}.csv")

    all_metrics: List[Dict[str, object]] = []

    print(f"Executing {len(ordered_jobs)} job(s). simulate={simulate}")
    for job in ordered_jobs:
        print(f"[EXEC] job_id={job.job_id} n={job.n} p={job.p}")
        metrics = execute_job(job, simulate=simulate)
        all_metrics.append(metrics)
        print(
            f"[DONE] job_id={job.job_id} duration_s={metrics['duration_s']} success={metrics['success']} exit={metrics['exit_code']}"
        )

    write_jsonl(all_metrics, jsonl_path)
    write_csv(all_metrics, csv_path)
    print(f"Wrote metrics: {jsonl_path}\nWrote metrics: {csv_path}")


def run_fifo_executor(args: argparse.Namespace) -> None:
    generate_jobs_if_requested(args)

    registry_jobs = collect_registry_jobs()
    if not registry_jobs:
        print("No jobs available to execute. Provide --generate-jobs or create jobs beforehand.")
        return

    ordered_jobs = fifo_order(registry_jobs)
    execute_ordered_jobs(ordered_jobs, output_dir=args.output_dir, simulate=args.simulate)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FIFO executor for Jobs with metrics output")
    parser.add_argument(
        "--generate-jobs",
        type=int,
        default=0,
        help="If > 0, generate this many jobs before executing",
    )
    parser.add_argument("--n-min", type=int, default=1, help="Minimum CPUs for generated jobs")
    parser.add_argument("--n-max", type=int, default=4, help="Maximum CPUs for generated jobs")
    parser.add_argument("--p-min", type=int, default=1, help="Minimum seconds for generated jobs")
    parser.add_argument("--p-max", type=int, default=10, help="Maximum seconds for generated jobs")

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate execution instead of running stress-ng",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to write metrics files",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_fifo_executor(args)


if __name__ == "__main__":
    main()



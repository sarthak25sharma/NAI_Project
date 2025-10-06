import random
import json
from datetime import datetime, timezone
from pathlib import Path
from jobs.job import Job  # import your Job class

class JobProducer:
    def __init__(self, n_range=(1, 4), p_range=(5, 30), max_jobs=10):
        """
        Initialize a job producer.
        - output_dir: directory where job JSON files will be saved
        - n_range: tuple -> (min_cpu, max_cpu) for stress-ng
        - p_range: tuple -> (min_sec, max_sec) for stress-ng runtime
        - max_jobs: number of jobs to generate
        """
        self.n_range = n_range
        self.p_range = p_range
        self.max_jobs = max_jobs

    def _generate_stress_command(self, n, p):
        """
        Generate a stress-ng command for CPU-bound job.
        Example: stress-ng --cpu 2 --timeout 10s
        """
        return f"stress-ng --cpu {n} --timeout {p}s"
    def generate_fucntion_signature(self,n,p):
        return "function of {n} iterations and {p} power "

    def produce_jobs(self):
        """Generate random jobs and save them as JSON files."""
        for job_id in range(1, self.max_jobs + 1):
            n = random.randint(*self.n_range)
            p = random.randint(*self.p_range)

            creation_time = datetime.now(timezone.utc)
            arrival_time = -1  # will be measured at cpu simulaotr 
            stress_command = self._generate_stress_command(n, p)
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
            ## add api here 
            
if __name__ == "__main__":
    # Example usage:
    producer = JobProducer(
        
        n_range=(1, 1000),
        p_range=(1,10),
        max_jobs=10
    )
    producer.produce_jobs()

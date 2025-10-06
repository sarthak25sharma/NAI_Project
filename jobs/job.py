from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json

@dataclass
class Job:
    job_id: int
    n: int 
    p:int 
    creation_time: datetime
    completed: bool
    completion_time: datetime | None
    arrival_time: datetime | None
    stress_command: str | None = None

    # Class-level registry to track all jobs by ID
    registry = {}

    def __post_init__(self):
        """Automatically register job instance by ID."""
        Job.registry[self.job_id] = self

    def mark_completed(self):
        """Mark this job as completed and record completion time."""
        self.completed = True
        self.completion_time = datetime.now(timezone.utc)


    @classmethod
    def get_job(cls, job_id: int):
        """Retrieve job instance by ID."""
        return cls.registry.get(job_id)

    @classmethod
    def mark_completed_by_id(cls, job_id: int):
        """Mark job as completed using job_id."""
        job = cls.get_job(job_id)
        if not job:
            raise ValueError(f"Job with ID {job_id} not found")
        job.mark_completed()
        return job

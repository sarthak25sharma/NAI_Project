# redis_test.py
import redis
import json

# Step 1: Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Step 2: Clear the queue (for testing)
r.delete("job_queue")

# Step 3: Push sample jobs
jobs = [
    {"id": 1, "score": 0.5, "data": "job-1"},
    {"id": 2, "score": 0.9, "data": "job-2"},
    {"id": 3, "score": 0.3, "data": "job-3"},
    {"id": 4, "score": 0.7, "data": "job-4"},
    {"id": 5, "score": 0.6, "data": "job-5"},
]

for job in jobs:
    r.rpush("job_queue", json.dumps(job))

print(f"Pushed {len(jobs)} jobs into Redis queue.")

# Step 4: Peek top k jobs
k = 3
top_k = []
for i in range(min(k, r.llen("job_queue"))):
    raw = r.lindex("job_queue", i)
    job = json.loads(raw)
    top_k.append((i, job))

print(f"\nTop {k} jobs:")
for idx, job in top_k:
    print(f"Index {idx}: {job}")

# Step 5: Select best job (highest score)
best_index, best_job = max(top_k, key=lambda x: x[1]["score"])
print(f"\nSelected best job: {best_job}")

# Step 6: Remove selected job safely
r.lset("job_queue", best_index, "__TO_DELETE__")
r.lrem("job_queue", 1, "__TO_DELETE__")

print(f"\nRemoved job at index {best_index}. Remaining jobs:")
for i in range(r.llen("job_queue")):
    raw = r.lindex("job_queue", i)
    job = json.loads(raw)
    print(f"Index {i}: {job}")
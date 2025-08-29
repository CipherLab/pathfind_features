from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Any, List


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StepJob:
    id: str
    created_at: float
    status: str  # PENDING|RUNNING|SUCCESS|ERROR|CANCELLED
    args: List[str]
    cwd: str
    logs_path: Path
    outputs: Dict[str, Any]
    return_code: Optional[int] = None

    def to_dict(self):
        d = asdict(self)
        d["logs_path"] = str(self.logs_path)
        return d


class StepJobs:
    def __init__(self):
        self._jobs: Dict[str, StepJob] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def start(self, args: List[str], logs_path: Path, outputs: Optional[Dict[str, Any]] = None, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> StepJob:
        job_id = str(uuid.uuid4())
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        job = StepJob(
            id=job_id,
            created_at=time.time(),
            status="PENDING",
            args=args,
            cwd=str(cwd or ROOT),
            logs_path=logs_path,
            outputs=outputs or {},
        )
        with self._global_lock:
            self._jobs[job_id] = job
            self._locks[job_id] = threading.Lock()
        # Start background thread
        t = threading.Thread(target=self._run, args=(job_id, env), daemon=True)
        t.start()
        return job

    def _run(self, job_id: str, env: Optional[Dict[str, str]]):
        with self._locks[job_id]:
            job = self._jobs[job_id]
            job.status = "RUNNING"
            # Ensure PYTHONPATH contains repo root
            run_env = os.environ.copy()
            if env:
                run_env.update(env)
            run_env["PYTHONPATH"] = str(ROOT) + (os.pathsep + run_env["PYTHONPATH"] if run_env.get("PYTHONPATH") else "")
            # Launch process with logs redirected to file
            with open(job.logs_path, "a", buffering=1) as logf:
                logf.write("Running: " + " ".join(job.args) + "\n")
                try:
                    proc = subprocess.Popen(job.args, cwd=job.cwd, stdout=logf, stderr=subprocess.STDOUT, env=run_env)
                    self._procs[job_id] = proc
                    code = proc.wait()
                    job.return_code = code
                    job.status = "SUCCESS" if code == 0 else "ERROR"
                except Exception as ex:
                    job.return_code = -1
                    job.status = "ERROR"
                    try:
                        logf.write(f"ERROR: {ex}\n")
                    except Exception:
                        pass
                finally:
                    self._jobs[job_id] = job

    def get(self, job_id: str) -> Optional[dict]:
        with self._global_lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def list(self) -> List[dict]:
        with self._global_lock:
            return [j.to_dict() for j in self._jobs.values()]

    def tail_logs(self, job_id: str, tail: int = 4000) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "not found"}
        if not job.logs_path.exists():
            return {"content": ""}
        txt = job.logs_path.read_text(encoding="utf-8", errors="ignore")
        return {"content": txt[-tail:]}

    def cancel(self, job_id: str) -> bool:
        proc = self._procs.get(job_id)
        if not proc:
            return False
        try:
            proc.terminate()
            return True
        except Exception:
            return False


STEP_JOBS = StepJobs()

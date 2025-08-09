from __future__ import annotations
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict

from .models import RunRequest, RunRecord

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "pipeline_runs"

class RunManager:
    def __init__(self):
        self._runs: Dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def list_runs(self):
        with self._lock:
            return [r.to_dict() for r in self._runs.values()]

    def list_runs_fs(self):
        """Scan pipeline_runs directory for historical runs and return minimal metadata."""
        runs = []
        if not RUNS_DIR.exists():
            return runs
        for p in sorted(RUNS_DIR.iterdir() if RUNS_DIR.exists() else [], key=lambda x: x.name):
            if not p.is_dir():
                continue
            summary = p / "run_summary.json"
            status = "UNKNOWN"
            try:
                if summary.exists():
                    import json
                    data = json.loads(summary.read_text(encoding="utf-8"))
                    status = data.get("status", "UNKNOWN")
            except Exception:
                pass
            runs.append({
                "name": p.name,
                "path": str(p),
                "status": status,
                "summary_path": str(summary) if summary.exists() else None,
                "logs_path": str(p / "logs.log"),
            })
        return runs

    def get(self, run_id: str):
        with self._lock:
            r = self._runs.get(run_id)
            return r.to_dict() if r else None

    def start(self, req: RunRequest) -> RunRecord:
        run_id = str(uuid.uuid4())
        timestamp = time.time()
        # Create run dir now so we can stream logs there
        run_suffix = req.run_name or "api"
        run_dir = RUNS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{run_suffix}_{run_id[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logs_path = run_dir / "logs.log"
        record = RunRecord(
            id=run_id,
            created_at=timestamp,
            status="PENDING",
            params=req,
            run_dir=run_dir,
            logs_path=logs_path,
        )
        with self._lock:
            self._runs[run_id] = record
        threading.Thread(target=self._run_thread, args=(run_id,), daemon=True).start()
        return record

    def _run_thread(self, run_id: str):
        with self._lock:
            record = self._runs[run_id]
            record.status = "RUNNING"
        args = [
            str((ROOT / ".venv/bin/python") if (ROOT/".venv/bin/python").exists() else sys.executable),
            str(ROOT / "run_pipeline.py"),
            "run",
            "--input-data", record.params.input_data,
            "--features-json", record.params.features_json,
            "--run-name", record.params.run_name or f"api_{run_id[:6]}",
        ]
        if getattr(record.params, 'stage1_from', None):
            args += ["--stage1-from", str(record.params.stage1_from)]
        if getattr(record.params, 'stage2_from', None):
            args += ["--stage2-from", str(record.params.stage2_from)]
        if record.params.force: args.append("--force")
        if record.params.skip_walk_forward: args.append("--skip-walk-forward")
        args += ["--max-new-features", str(record.params.max_new_features)]
        if record.params.yolo_mode: args.append("--yolo-mode")
        if record.params.pf_debug: args.append("--pf-debug")
        args += ["--pf-debug-every-rows", str(record.params.pf_debug_every_rows)]
        if record.params.disable_pathfinding: args.append("--disable-pathfinding")
        if record.params.pretty: args.append("--pretty")
        if record.params.smoke_mode: args.append("--smoke-mode")
        if record.params.smoke_max_eras is not None:
            args += ["--smoke-max-eras", str(record.params.smoke_max_eras)]
        if record.params.smoke_row_limit is not None:
            args += ["--smoke-row-limit", str(record.params.smoke_row_limit)]
        if record.params.smoke_feature_limit is not None:
            args += ["--smoke-feature-limit", str(record.params.smoke_feature_limit)]
        args += ["--seed", str(record.params.seed)]

        # Run process, log to file and capture result
        with open(record.logs_path, "a", buffering=1) as logf:
            logf.write("Running: " + " ".join(args) + "\n")
            try:
                proc = subprocess.run(args, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)
                code = proc.returncode
            except Exception as e:
                code = -1
                logf.write(f"ERROR: {e}\n")
        with self._lock:
            record = self._runs[run_id]
            if code == 0:
                record.status = "SUCCESS"
            else:
                record.status = "ERROR"
                record.error = f"exit={code}"
            # Save summary if present
            summary = record.run_dir / "run_summary.json"
            if summary.exists():
                # Patch-in phase and lineage for UI if not present
                try:
                    import json
                    data = json.loads(summary.read_text(encoding="utf-8"))
                    data.setdefault("ui_meta", {})
                    if getattr(record.params, "phase", None):
                        data["ui_meta"]["phase"] = record.params.phase
                    # Attempt to infer lineage from reused stages
                    lineage = {}
                    if getattr(record.params, "stage1_from", None):
                        lineage["stage1_from"] = str(record.params.stage1_from)
                    if getattr(record.params, "stage2_from", None):
                        lineage["stage2_from"] = str(record.params.stage2_from)
                    if lineage:
                        data["ui_meta"]["inheritance"] = lineage
                    summary.write_text(json.dumps(data, indent=2))
                except Exception:
                    pass
                record.result = {"summary_path": str(summary)}
            self._runs[run_id] = record

RUNS = RunManager()

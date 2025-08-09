from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Any
from pathlib import Path

RunStatus = Literal["PENDING", "RUNNING", "SUCCESS", "ERROR", "CANCELLED"]

Phase = Literal["full", "target", "pathfinding", "features"]


@dataclass
class RunRequest:
    input_data: str
    features_json: str
    run_name: Optional[str] = None
    stage1_from: Optional[str] = None
    stage2_from: Optional[str] = None
    # Optional phase hint: controls UI semantics; runner still invokes unified pipeline
    phase: Optional[Phase] = None
    force: bool = False
    skip_walk_forward: bool = False
    max_new_features: int = 20
    yolo_mode: bool = False
    pf_debug: bool = False
    pf_debug_every_rows: int = 10000
    disable_pathfinding: bool = False
    pretty: bool = True
    smoke_mode: bool = False
    smoke_max_eras: Optional[int] = None
    smoke_row_limit: Optional[int] = None
    smoke_feature_limit: Optional[int] = None
    seed: int = 42

@dataclass
class RunRecord:
    id: str
    created_at: float
    status: RunStatus
    params: RunRequest
    run_dir: Path
    logs_path: Path
    result: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        d["run_dir"] = str(self.run_dir)
        d["logs_path"] = str(self.logs_path)
        return d

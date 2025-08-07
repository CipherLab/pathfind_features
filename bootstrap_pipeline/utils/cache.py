import os
import json
import hashlib
import shutil
import logging


def _file_fingerprint(path: str):
    try:
        stat = os.stat(path)
        return {"path": path, "mtime": stat.st_mtime, "size": stat.st_size}
    except FileNotFoundError:
        return {"path": path, "missing": True}


def compute_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode()).hexdigest()  # noqa: S324 (not for security)


def stage_cache_lookup(stage: str, hash_key: str, cache_root: str = ".cache"):
    stage_dir = os.path.join(cache_root, stage, hash_key)
    meta_path = os.path.join(stage_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return stage_dir, meta
        except Exception:
            return None, None
    return None, None


def stage_cache_store(stage: str, hash_key: str, artifacts: dict, params: dict, cache_root: str = ".cache"):
    stage_dir = os.path.join(cache_root, stage, hash_key)
    os.makedirs(stage_dir, exist_ok=True)
    meta = {"params": params, "artifacts": artifacts}
    with open(os.path.join(stage_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    # Copy artifacts into cache directory
    for label, path in artifacts.items():
        if not os.path.isfile(path):
            continue
        try:
            shutil.copy2(path, os.path.join(stage_dir, os.path.basename(path)))
        except Exception as e:  # pragma: no cover
            logging.warning(f"Cache copy failed for {path}: {e}")
    return stage_dir


def materialize_cached_artifacts(stage_dir: str, artifacts: dict, destination_dir: str):
    for label, original_path in artifacts.items():
        filename = os.path.basename(original_path)
        cached_path = os.path.join(stage_dir, filename)
        dest_path = original_path
        if os.path.exists(cached_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            try:
                shutil.copy2(cached_path, dest_path)
            except Exception as e:
                logging.warning(f"Failed to restore cached artifact {filename}: {e}")

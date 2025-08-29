import numpy as np
from tests import setup_script_output, get_output_path, initialize_script_output, add_output_dir_arguments

__all__ = [
    "era_sanity",
    "safe_sharpe",
    "feature_condition_number",
]

def era_sanity(x, atol: float = 1e-12):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0, "n_unique": 0, "range": 0.0, "std": 0.0, "mad": 0.0, "is_constant": True}
    nuniq = np.unique(np.round(x, 12)).size
    xr = float(np.max(x) - np.min(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    mad = float(np.median(np.abs(x - np.median(x)))) if x.size > 0 else 0.0
    return {
        "n": int(len(x)),
        "n_unique": int(nuniq),
        "range": xr,
        "std": std,
        "mad": mad,
        "is_constant": (nuniq <= 1) or (xr < atol) or (std < atol) or (mad < atol),
    }


def safe_sharpe(x, eps: float = 1e-9):
    x = np.asarray(x, dtype=np.float64)
    chk = era_sanity(x)
    if chk.get("is_constant", False):
        return np.nan, {"reason": "degenerate", **chk, "mean": float(np.mean(x)) if x.size else 0.0, "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0}
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if sd < eps:
        mad = float(np.median(np.abs(x - np.median(x)))) if x.size > 0 else 0.0
        if mad < eps:
            return np.nan, {"reason": "zero_var_and_mad", **chk, "mean": mu, "std": sd}
        sd = 1.4826 * mad
        return mu / (sd + eps), {"reason": "robust", **chk, "mean": mu, "std": sd}
    return mu / (sd + eps), {"reason": "ok", **chk, "mean": mu, "std": sd}


def feature_condition_number(X, eps: float = 1e-10, shrink: float = 0.0):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0:
        return 1.0, 1.0
    # Use correlation to reduce scale issues; compute in chunks to reduce memory if features are huge
    # For 2000 columns this is fine; for >10k, prefer online covariance
    C = np.corrcoef(X, rowvar=False)
    if not np.all(np.isfinite(C)):
        C = np.nan_to_num(C, nan=0.0)
    if shrink > 0.0:
        p = C.shape[0]
        C = (1.0 - shrink) * C + shrink * np.eye(p, dtype=np.float64)
    w = np.linalg.eigvalsh(C)
    w = np.maximum(w, eps)
    return float(w[-1] / w[0]), float(w[0])

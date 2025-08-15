# bootstrap_pipeline/bootstrap/target_discovery.py

import logging
import random
import os
from collections import OrderedDict
import hashlib
import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from .metrics_utils import era_sanity, safe_sharpe, feature_condition_number

try:
    # Optional: for fast linear algebra (falls back to numpy if not present)
    import numpy.linalg as npl
except Exception:  # pragma: no cover
    npl = None

class WalkForwardTargetDiscovery:
    """Walk-forward target weight discovery with multiple evaluation modes for speed."""

    def __init__(
        self,
        target_columns,
        min_history_eras: int = 20,
        eval_mode: str | None = None,
        top_full_models: int = 3,
        ridge_lambda: float = 1.0,
        sample_per_era: int = 2000,
        max_combinations: int = 15,
        feature_fraction: float = 0.8,
        random_state: int = 42,
        num_boost_round: int = 12,
        max_era_cache: int = 0,
        clear_cache_every: int = 0,
        pre_cache_dir: str | None = None,
        persist_pre_cache: bool = False,
        # New knobs (opt-in; defaults keep legacy behavior)
        use_tversky: bool = False,
        tversky_k: int = 8,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
    robust_stats: bool = True,
    # Era quality controls
    skip_degenerate_eras: bool = False,
    mad_tol: float = 1e-12,
    ) -> None:
        # Core config
        self.target_columns = target_columns
        self.n_targets = len(target_columns)
        self.min_history_eras = min_history_eras
        # State
        self.era_weights = {}
        self.last_weights: np.ndarray | None = None
        # Hyperparameters / knobs
        self.SIGN_CONSISTENCY_OFFSET = 0.5
        self.top_full_models = top_full_models
        self.ridge_lambda = ridge_lambda
        self.sample_per_era = sample_per_era
        self.max_combinations = max_combinations
        self.feature_fraction = feature_fraction
        self.random_state = random_state
        self.num_boost_round = num_boost_round
        self.max_era_cache = max_era_cache  # 0 => unlimited
        self.clear_cache_every = clear_cache_every  # 0 => never
        self.pre_cache_dir = pre_cache_dir or os.environ.get("TD_PRE_CACHE_DIR", "cache/td_pre_cache")
        self.persist_pre_cache = persist_pre_cache
        # New feature projection / robustness knobs
        self.use_tversky = bool(use_tversky or (os.environ.get("TD_USE_TVERSKY", "0") == "1"))
        self.tversky_k = int(os.environ.get("TD_TVERSKY_K", tversky_k))
        self.tversky_alpha = float(os.environ.get("TD_TVERSKY_ALPHA", tversky_alpha))
        self.tversky_beta = float(os.environ.get("TD_TVERSKY_BETA", tversky_beta))
        self.robust_stats = robust_stats
        # Era quality controls
        self.skip_degenerate_eras = bool(skip_degenerate_eras or (os.environ.get("TD_SKIP_DEGENERATE_ERAS", "0") == "1"))
        self.mad_tol = float(os.environ.get("TD_MAD_TOL", mad_tol))
        if self.persist_pre_cache:
            os.makedirs(self.pre_cache_dir, exist_ok=True)
        # Era-level preprocessing cache (LRU OrderedDict: era -> preprocessed data)
        self._era_pre_cache = OrderedDict()

        if eval_mode is None:
            eval_mode = os.environ.get("TD_EVAL_MODE", "hybrid")
        self.eval_mode = eval_mode.lower()
        if self.eval_mode not in {"gbm_full", "linear_fast", "hybrid"}:
            logging.warning("Unknown eval_mode %s; defaulting to hybrid", self.eval_mode)
            self.eval_mode = "hybrid"

        logging.info(
            "Initialized walk-forward target discovery: targets=%s, mode=%s, top_full=%s, cache_limit=%s",
            self.n_targets,
            self.eval_mode,
            self.top_full_models,
            self.max_era_cache or '∞',
        )

    # ---------------- Robust stats + diagnostics helpers -----------------
    # moved to metrics_utils: era_sanity, safe_sharpe, feature_condition_number

    # ---------------- Tversky projection helpers (optional) ---------------
    @staticmethod
    def _zpos_binarize(X, mu=None, sd=None):
        X = np.asarray(X, dtype=np.float64)
        if mu is None or sd is None:
            mu = X.mean(0)
            sd = X.std(0, ddof=1)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return ( (X - mu) / sd > 0.0 ), (mu, sd)

    @staticmethod
    def _k_medoids_indices(X, k, seed=0):
        """Memory-safe farthest-first prototype selection (no O(n^2) matrix).

        Picks a random start, then greedily selects the point with max min-distance
        to the current set. Approximate but O(n*k*p) memory and works for large n.
        """
        rng = np.random.default_rng(seed)
        n = len(X)
        if n == 0:
            return np.array([], dtype=int)
        k = max(1, min(int(k), n))
        # start from a random index
        centers = [int(rng.integers(0, n))]
        # maintain min distances to chosen centers
        min_d = np.full(n, np.inf, dtype=np.float64)
        x0 = X[centers[0]]
        min_d = np.minimum(min_d, np.linalg.norm(X - x0[None, :], axis=1))
        for _ in range(1, k):
            j = int(np.argmax(min_d))
            centers.append(j)
            xj = X[j]
            dj = np.linalg.norm(X - xj[None, :], axis=1)
            min_d = np.minimum(min_d, dj)
        return np.array(centers, dtype=int)

    @staticmethod
    def _tversky_bool(A, B, alpha=0.7, beta=0.3, theta=1.0):
        inter = (A & B).sum(-1)
        a_only = (A & ~B).sum(-1)
        b_only = (B & ~A).sum(-1)
        return theta * inter - alpha * a_only - beta * b_only

    @classmethod
    def _tversky_projection_features(cls, X_tr, X_te, k=8, alpha=0.7, beta=0.3, seed=0):
        A_tr, stats = cls._zpos_binarize(X_tr)
        mu, sd = stats
        A_te, _ = cls._zpos_binarize(X_te, mu=mu, sd=sd)
        idx = cls._k_medoids_indices(X_tr, k, seed=seed)
        if idx.size == 0:
            return (np.zeros((X_tr.shape[0], 0), dtype=np.float32), np.zeros((X_te.shape[0], 0), dtype=np.float32),
                    {"mu": mu, "sd": sd, "proto_idx": idx, "alpha": alpha, "beta": beta})
        P = A_tr[idx]  # boolean prototypes
        # Preallocate to reduce temporaries
        m_tr, m_te, kP = A_tr.shape[0], A_te.shape[0], P.shape[0]
        S_tr = np.empty((m_tr, kP), dtype=np.int32)
        S_te = np.empty((m_te, kP), dtype=np.int32)
        for j in range(kP):
            s_tr_col = cls._tversky_bool(A_tr, P[j], alpha, beta)
            s_te_col = cls._tversky_bool(A_te, P[j], alpha, beta)
            S_tr[:, j] = s_tr_col
            S_te[:, j] = s_te_col
        meta = {"mu": mu, "sd": sd, "proto_idx": idx, "alpha": alpha, "beta": beta}
        return S_tr.astype(np.float32), S_te.astype(np.float32), meta

    def generate_smart_combinations(self, n_combinations: int | None = None) -> np.ndarray:
        """Generate target combinations to test - fewer but smarter"""
        if n_combinations is None:
            n_combinations = self.max_combinations
        combinations = []
        
        # All pure single-target combinations
        for i in range(self.n_targets):
            weights = np.zeros(self.n_targets)
            weights[i] = 1.0
            combinations.append(weights)
        
        # Equal weight (the baseline everyone uses)
        combinations.append(np.ones(self.n_targets) / self.n_targets)
        
        # Top-heavy (first few indices for now; later: rank by history)
        for focus in [2, 3]:
            if focus <= self.n_targets:
                weights = np.zeros(self.n_targets)
                weights[:focus] = np.random.dirichlet(np.ones(focus))
                combinations.append(weights)
        
        # Random sparse combinations (respect cap after base set)
        base_len = len(combinations)
        if n_combinations < base_len:
            # Prune if caller requested fewer than base (keep most interpretable first entries)
            return np.array(combinations[:n_combinations])
        remaining = max(0, n_combinations - base_len)
        for _ in range(remaining):
            weights = np.zeros(self.n_targets)
            n_active = random.randint(2, min(4, self.n_targets))
            active_idx = random.sample(range(self.n_targets), n_active)
            active_weights = np.random.dirichlet(np.ones(n_active))
            for i, idx in enumerate(active_idx):
                weights[idx] = active_weights[i]
            combinations.append(weights)
        
        return np.array(combinations)
    
    def evaluate_combination_robustly(self, weights, history_df, feature_cols):
        """
        Evaluate a target combination across multiple historical eras
        Uses cross-validation instead of cherry-picking the best era
        """
        try:
            eras = history_df['era'].unique()
            if len(eras) < 3:
                return 0.0, 0.0, 0.0, 0.0

            # Build / reuse preprocessing cache per era
            rng = np.random.default_rng(self.random_state)
            era_scores_raw: list[float] = []

            # Training params (keep light but meaningful)
            # LightGBM knobs (lazy trees, more regularized by default)
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'max_depth': 6,
                'num_leaves': 63,
                'learning_rate': 0.05,
                'feature_fraction': max(0.05, min(float(self.feature_fraction), 1.0)),
                'bagging_fraction': 0.7,
                'bagging_freq': 1,
                'max_bin': 63,
                'verbosity': -1,
                'seed': int(self.random_state),
                'num_threads': int(os.environ.get('TD_NUM_THREADS', '-1')),
            }
            if os.environ.get("TD_USE_GPU") == "1":
                params['device_type'] = 'gpu'
                if 'TD_GPU_PLATFORM_ID' in os.environ:
                    params['gpu_platform_id'] = int(os.environ['TD_GPU_PLATFORM_ID'])
                if 'TD_GPU_DEVICE_ID' in os.environ:
                    params['gpu_device_id'] = int(os.environ['TD_GPU_DEVICE_ID'])

            for era in eras:
                era_data = history_df[history_df['era'] == era]
                if len(era_data) < 100:
                    continue
                cache = self._load_or_build_era_cache(era, era_data, feature_cols, rng)
                if cache is None:
                    continue

                # Compose labels for this weight vector without rebuilding features
                y_tr = cache['Y_tr_mat'] @ weights
                y_te = cache['Y_te_mat'] @ weights
                # Sanity check labels; skip degenerate eras loudly
                lbl_chk = era_sanity(y_tr)
                if lbl_chk.get("is_constant", False):
                    logging.warning("[degenerate-era] era=%s labels constant-ish; skip Sharpe/IR. stats=%s", era, lbl_chk)
                    continue
                # Skip degenerate labels
                if np.allclose(np.std(y_tr), 0):
                    continue

                # Reuse dataset and set label per combo
                train_set = cache['train_set']
                # Min data in leaf scaled by training rows
                try:
                    n_rows = len(y_tr)
                    params['min_data_in_leaf'] = max(20, int(0.03 * n_rows))
                except Exception:
                    params['min_data_in_leaf'] = 50
                train_set.set_label(y_tr)
                model = lgb.train(
                    params,
                    train_set,
                    num_boost_round=int(self.num_boost_round),
                    callbacks=[lgb.log_evaluation(0)],
                )
                preds = model.predict(cache['X_te'])
                # Correlation between preds and target for this era
                p = np.asarray(preds, dtype=np.float64)
                t = np.asarray(y_te, dtype=np.float64)
                # Era diagnostics: predictions
                pred_chk = era_sanity(p)
                if pred_chk.get("is_constant", False):
                    logging.warning("[degenerate-preds] era=%s predictions constant-ish; skip. stats=%s", era, pred_chk)
                    continue
                p -= p.mean()
                t -= t.mean()
                denom = (np.sqrt((p**2).sum()) * np.sqrt((t**2).sum()))
                if denom == 0 or not np.isfinite(denom):
                    continue
                corr = float((p * t).sum() / denom)
                if np.isfinite(corr):
                    era_scores_raw.append(corr)

            if not era_scores_raw:
                return 0.0, 0.0, 0.0, 0.0

            signs = [c > 0 for c in era_scores_raw]
            sign_consistency = float(np.mean(signs)) if signs else 0.0
            # Robust Sharpe across eras
            sharpe, meta = safe_sharpe(era_scores_raw) if self.robust_stats else (
                float(np.mean(era_scores_raw)) / (float(np.std(era_scores_raw)) + 1e-6), {"reason": "legacy"}
            )
            mean_score = float(np.abs(np.mean(era_scores_raw)))
            std_score = float(meta.get("std", np.std(era_scores_raw)))
            if not np.isfinite(sharpe):
                logging.warning("Sharpe ratio non-finite across eras; meta=%s", meta)
                sharpe = 0.0
            return mean_score, std_score, sign_consistency, float(sharpe)
        except Exception as e:
            logging.warning(f"Evaluation error: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def _prepare_history_matrices(self, history_df, feature_cols):
        """Prepare per-era sampled & imputed train/test matrices reused across combos.
                        model = lgb.train(
                            params,
                            cache['train_set'],
                            num_boost_round=self.num_boost_round,
                            callbacks=[lgb.log_evaluation(0)],
                        )
        'weights_helper': (M), 'era': era }
        Where pred_basis = X_test @ M with M = (X^T X + lambda I)^-1 X^T Y (ridge solution)
        so predictions for any weight vector w is (pred_basis @ w).
        """
        rng = np.random.default_rng(self.random_state)
        out = []
        eras = history_df['era'].unique()
        # (Optional) feature subsampling for speed
        feature_cols_used = feature_cols
        if 0 < self.feature_fraction < 1.0:
            k = max(1, int(len(feature_cols) * self.feature_fraction))
            feature_cols_used = rng.choice(feature_cols, size=k, replace=False)

        for era in eras:
            era_data = history_df[history_df['era'] == era]
            if len(era_data) < 100:  # skip tiny eras
                continue
            if len(era_data) > self.sample_per_era:
                era_data = era_data.sample(n=self.sample_per_era, random_state=self.random_state)

            X = era_data[feature_cols_used].to_numpy(dtype=float, copy=False)
            Y = era_data[self.target_columns].to_numpy(dtype=float, copy=False)

            # Era quality skip: if all targets have negligible MAD, skip this era
            if self.skip_degenerate_eras:
                try:
                    mads = np.median(np.abs(Y - np.median(Y, axis=0, keepdims=True)), axis=0)
                    if np.all(mads < self.mad_tol):
                        logging.info("[skip-era] era=%s skipped due to low MAD across all targets (min_mad=%.3e, tol=%.3e)", era, float(np.min(mads)), self.mad_tol)
                        continue
                except Exception:
                    pass

            # Simple median imputation
            med = np.nanmedian(X, axis=0)
            med = np.where(np.isnan(med), 0.0, med)
            X = np.where(np.isnan(X), med, X)
            tmed = np.nanmedian(Y, axis=0)
            tmed = np.where(np.isnan(tmed), 0.0, tmed)
            Y = np.where(np.isnan(Y), tmed, Y)

            if X.shape[0] < 20:
                continue
            # Deterministic split 70/30
            n = X.shape[0]
            idx = np.arange(n)
            rng.shuffle(idx)
            cut = int(n * 0.7)
            train_idx, test_idx = idx[:cut], idx[cut:]
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Ridge precomputation
            # Ridge precomputation with centering: (Xc^T Xc + λI)^-1 Xc^T Yc, pred = (Xtest - X_mean) M + Y_mean
            X_mean = X_train.mean(axis=0, keepdims=True)
            Y_mean = Y_train.mean(axis=0, keepdims=True)
            Xc = X_train - X_mean
            Yc = Y_train - Y_mean
            # Conditioning diagnostics
            cond, min_eig = feature_condition_number(X_train)
            applied_shrink = 0.0
            if cond > 1e12 or min_eig < 1e-6:
                applied_shrink = 0.1
                logging.info("[cond] era=%s cond=%.2e min_eig=%.2e -> diagonal loading=%.2f", era, cond, min_eig, applied_shrink)
            XtX = Xc.T @ Xc
            lam = self.ridge_lambda * (10.0 if (cond > 1e12 or min_eig < 1e-6) else 1.0)
            XtX.ravel()[:: XtX.shape[0] + 1] += lam
            XtY = Xc.T @ Yc
            try:
                M = np.linalg.solve(XtX, XtY)
            except Exception:
                M = np.linalg.pinv(XtX) @ XtY
            pred_basis = (X_test - X_mean) @ M + Y_mean  # shape (n_test, n_targets)
            out.append({
                'era': era,
                'pred_basis': pred_basis,
                'Y_test': Y_test,
                'cond': cond,
                'min_eig': min_eig,
                'applied_shrink': applied_shrink,
            })
        return out

    def _score_combinations_linear(self, era_mats, combinations):
        """Vectorized scoring of combinations using precomputed linear prediction bases.

        For each era: we have pred_basis (n_test, n_targets) and Y_test (n_test, n_targets).
        For all combinations W (n_combos, n_targets): preds = P @ W^T, ys = Y @ W^T.
        We compute correlations column-wise and then aggregate across eras.
        """
        if len(era_mats) == 0:
            return []
        W = np.asarray(combinations, dtype=np.float32)  # (n_combos, n_targets)
        n_combos = W.shape[0]
        corr_list = []  # list of arrays shape (n_combos,)
        for em in era_mats:
            P = em['pred_basis'].astype(np.float32, copy=False)  # (n_test, n_targets)
            Y = em['Y_test'].astype(np.float32, copy=False)
            if P.shape[0] < 3:
                continue
            preds = P @ W.T  # (n_test, n_combos)
            ys = Y @ W.T
            # Compute correlation per combo vectorized
            preds_mean = preds.mean(axis=0)
            ys_mean = ys.mean(axis=0)
            preds_center = preds - preds_mean
            ys_center = ys - ys_mean
            denom = (np.sqrt((preds_center**2).sum(axis=0)) * np.sqrt((ys_center**2).sum(axis=0)))
            # Avoid div zero
            with np.errstate(divide='ignore', invalid='ignore'):
                corr = (preds_center * ys_center).sum(axis=0) / denom
            corr = np.where(np.isfinite(corr), corr, np.nan)
            corr_list.append(corr)
        if not corr_list:
            return []
        C = np.vstack(corr_list)  # (n_valid_eras, n_combos)
        # Mask NaNs per combo
        valid_mask = np.isfinite(C)
        # Replace NaNs with 0 for stats where count >=2 else treat as insufficient
        results = []
        for j in range(n_combos):
            col = C[:, j]
            vm = valid_mask[:, j]
            valid_vals = col[vm]
            if valid_vals.size < 2:
                results.append({'weights': W[j], 'mean_score': 0.0, 'std_score': 0.0, 'sign_consistency': 0.0, 'sharpe': 0.0})
                continue
            signs = valid_vals > 0
            sign_consistency = float(signs.mean())
            # Robust Sharpe across eras for this combo
            sharpe, meta = safe_sharpe(valid_vals) if self.robust_stats else (
                float(np.mean(valid_vals)) / (float(np.std(valid_vals)) + 1e-6), {"reason": "legacy"}
            )
            mean_score = float(np.abs(np.mean(valid_vals)))
            std_score = float(meta.get("std", np.std(valid_vals)))
            if not np.isfinite(sharpe):
                sharpe = 0.0
            results.append({'weights': W[j], 'mean_score': mean_score, 'std_score': std_score, 'sign_consistency': sign_consistency, 'sharpe': float(sharpe)})
        return results

    def discover_weights_for_era(self, current_era, history_df, feature_cols):
        """Find the best target weights using ONLY historical data.

        Now supports multiple evaluation modes for speed.
        """
        if len(history_df) < 1000:
            return np.ones(self.n_targets) / self.n_targets

        combinations = self.generate_smart_combinations()

        if self.eval_mode == 'gbm_full':
            combination_results = []
            best_sharpe = -1e9
            stale = 0
            base_floor = self.n_targets + 3  # ensure we cover singles + equal + a few extras before early stop
            for idx, w in enumerate(combinations):
                ms, ss, sc, sh = self.evaluate_combination_robustly(w, history_df, feature_cols)
                combination_results.append({'weights': w, 'mean_score': ms, 'std_score': ss, 'sign_consistency': sc, 'sharpe': sh})
                if sh > best_sharpe + 1e-3:
                    best_sharpe = sh
                    stale = 0
                else:
                    stale += 1
                if stale >= 3 and idx >= base_floor:
                    logging.debug("Early stop in gbm_full after %s combos (no Sharpe improvement)", idx + 1)
                    break
        else:
            # Fast linear screening
            era_mats = self._prepare_history_matrices(history_df, feature_cols)
            combination_results = self._score_combinations_linear(era_mats, combinations)
            if self.eval_mode == 'hybrid':
                # Take top-K and refine with full LightGBM
                combination_results.sort(key=lambda x: x['sharpe'], reverse=True)
                top = combination_results[: self.top_full_models]
                refined = []
                for entry in top:
                    w = entry['weights']
                    ms, ss, sc, sh = self.evaluate_combination_robustly(w, history_df, feature_cols)
                    refined.append({'weights': w, 'mean_score': ms, 'std_score': ss, 'sign_consistency': sc, 'sharpe': sh})
                # Replace the top entries with refined values
                combination_results[: self.top_full_models] = refined

        # Sort final results
        combination_results.sort(key=lambda x: x['sharpe'], reverse=True)

        if not combination_results:
            return np.ones(self.n_targets) / self.n_targets

        best = combination_results[0]
        drift = 1.0
        if self.last_weights is not None:
            norm_best = np.linalg.norm(best['weights'])
            norm_last = np.linalg.norm(self.last_weights)
            denom = norm_best * norm_last
            if denom != 0:
                drift = np.dot(best['weights'], self.last_weights) / denom
            elif np.allclose(best['weights'], 0) and np.allclose(self.last_weights, 0):
                drift = 1.0
            else:
                drift = 0.0
        eff_targets_denom = np.sum(best['weights'] ** 2)
        if eff_targets_denom == 0:
            eff_targets = float('inf')
            logging.warning(
                f"EffTargets denominator is zero for era {current_era}. Setting EffTargets to infinity.")
        else:
            eff_targets = 1 / eff_targets_denom
        logging.info(
            f"Era {current_era}: Mean={best['mean_score']:.4f}, Sharpe={best['sharpe']:.3f}, "
            f"Sign+={best['sign_consistency']:.2%}, EffTargets={eff_targets:.2f}, Drift={drift:.3f} (mode={self.eval_mode})"
        )
        self.last_weights = best['weights']
        return best['weights']

    # ---------------- Cache management helpers -----------------
    def _maybe_trim_cache(self):
        if self.max_era_cache and len(self._era_pre_cache) > self.max_era_cache:
            # pop oldest
            popped_era, _ = self._era_pre_cache.popitem(last=False)
            logging.debug("Evicted era %s from preprocessing cache (max=%s)", popped_era, self.max_era_cache)

    def maybe_periodic_cache_clear(self, era_index: int):
        if self.clear_cache_every and (era_index + 1) % self.clear_cache_every == 0:
            self._era_pre_cache.clear()
            logging.info("Cleared preprocessing cache at era index %s (every %s eras)", era_index + 1, self.clear_cache_every)

    # ---------------- Persistent cache helpers -----------------
    def _era_cache_file(self, era: str, feature_cols) -> str:
        if not self.persist_pre_cache:
            return ""
        sig_payload = {
            'era': era,
            'targets': self.target_columns,
            'n_targets': self.n_targets,
            'sample_per_era': self.sample_per_era,
            'feature_fraction': self.feature_fraction,
            'random_state': self.random_state,
            'features': list(feature_cols),
            'ridge_lambda': self.ridge_lambda,
            'use_tversky': self.use_tversky,
            'tversky_k': self.tversky_k,
            'tversky_alpha': self.tversky_alpha,
            'tversky_beta': self.tversky_beta,
            'skip_degenerate_eras': self.skip_degenerate_eras,
            'mad_tol': self.mad_tol,
        }
        sig = hashlib.md5(json.dumps(sig_payload, sort_keys=True).encode('utf-8')).hexdigest()
        return os.path.join(self.pre_cache_dir, f"era_pre_{sig}.npz")

    def _load_or_build_era_cache(self, era: str, era_data, feature_cols, rng):
        # Try persistent load
        path = self._era_cache_file(era, feature_cols)
        if path and os.path.exists(path):
            try:
                d = np.load(path)
                cache = {
                    'train_set': lgb.Dataset(d['X_tr'], free_raw_data=False),
                    'X_te': d['X_te'],
                    'Y_tr_mat': d['Y_tr_mat'],
                    'Y_te_mat': d['Y_te_mat'],
                }
                self._era_pre_cache[era] = cache
                self._maybe_trim_cache()
                return cache
            except Exception:
                pass  # fallback to rebuild
        # Build anew
        if len(era_data) > self.sample_per_era:
            era_data = era_data.sample(n=self.sample_per_era, random_state=self.random_state)
        X = era_data[feature_cols].to_numpy(dtype=np.float32, copy=False)
        Ymat = era_data[self.target_columns].to_numpy(dtype=np.float32, copy=False)
        n = X.shape[0]
        if n < 50:
            return None
        # Era quality skip: evaluate MAD on targets; skip if all below tolerance
        if self.skip_degenerate_eras:
            try:
                mads = np.median(np.abs(Ymat - np.median(Ymat, axis=0, keepdims=True)), axis=0)
                if np.all(mads < self.mad_tol):
                    logging.info("[skip-era] era=%s skipped (cache build) due to low MAD across all targets (min_mad=%.3e, tol=%.3e)", era, float(np.min(mads)), self.mad_tol)
                    return None
            except Exception:
                pass
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(n * 0.7)
        tr_idx, te_idx = idx[:cut], idx[cut:]
        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr_mat, Y_te_mat = Ymat[tr_idx], Ymat[te_idx]
        med = np.nanmedian(X_tr, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        X_tr = np.where(np.isnan(X_tr), med, X_tr)
        X_te = np.where(np.isnan(X_te), med, X_te)
        # Optional: append Tversky projection features
        tversky_meta = None
        if self.use_tversky and X_tr.shape[1] > 0:
            try:
                S_tr, S_te, meta = self._tversky_projection_features(
                    X_tr.astype(np.float64, copy=False),
                    X_te.astype(np.float64, copy=False),
                    k=self.tversky_k,
                    alpha=self.tversky_alpha,
                    beta=self.tversky_beta,
                    seed=self.random_state,
                )
                if S_tr.shape[1] > 0:
                    X_tr = np.concatenate([X_tr, S_tr], axis=1)
                    X_te = np.concatenate([X_te, S_te], axis=1)
                    tversky_meta = meta
                    logging.debug("[tversky] era=%s added %s proto features (k=%s, a=%.2f, b=%.2f)", era, S_tr.shape[1], self.tversky_k, self.tversky_alpha, self.tversky_beta)
            except Exception as e:
                logging.warning("Tversky feature construction failed for era %s: %s", era, e)
        train_set = lgb.Dataset(X_tr, free_raw_data=False)
        cache = {
            'train_set': train_set,
            'X_te': X_te,
            'Y_tr_mat': Y_tr_mat,
            'Y_te_mat': Y_te_mat,
            'tversky_meta': tversky_meta,
        }
        self._era_pre_cache[era] = cache
        self._maybe_trim_cache()
        # Persist
        if path:
            try:
                np.savez_compressed(path, X_tr=X_tr, X_te=X_te, Y_tr_mat=Y_tr_mat, Y_te_mat=Y_te_mat)
            except Exception:
                pass
        return cache

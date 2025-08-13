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
    ) -> None:
        # Core config
        self.target_columns = target_columns
        self.n_targets = len(target_columns)
        self.min_history_eras = min_history_eras
        # State
        self.era_weights: dict = {}
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
        if self.persist_pre_cache:
            os.makedirs(self.pre_cache_dir, exist_ok=True)
        # Era-level preprocessing cache (LRU OrderedDict: era -> preprocessed data)
        self._era_pre_cache: OrderedDict[str, dict] = OrderedDict()

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
            era_scores_raw = []
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 15,
                'learning_rate': 0.12,
                'feature_fraction': 0.25,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'max_bin': 63,
                'min_data_in_leaf': 50,
                'verbosity': -1,
                'seed': self.random_state,
            }
            for era in eras:
                era_data = history_df[history_df['era'] == era]
                if len(era_data) < 100:
                    continue
                cache = self._load_or_build_era_cache(era, era_data, feature_cols, rng)
                if cache is None:
                    continue
                    params = {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'num_leaves': 15,
                        'learning_rate': 0.12,
                        'feature_fraction': max(0.05, min(self.feature_fraction, 1.0)),
                        'bagging_fraction': 0.8,
                        'bagging_freq': 1,
                        'max_bin': 63,
                        'min_data_in_leaf': 50,
                        'verbosity': -1,
                        'seed': self.random_state,
                        'num_threads': int(os.environ.get('TD_NUM_THREADS', '-1')),
                    }
                    if os.environ.get("TD_USE_GPU") == "1":
                        params['device_type'] = 'gpu'
                        if 'TD_GPU_PLATFORM_ID' in os.environ:
                            params['gpu_platform_id'] = int(os.environ['TD_GPU_PLATFORM_ID'])
                        if 'TD_GPU_DEVICE_ID' in os.environ:
                            params['gpu_device_id'] = int(os.environ['TD_GPU_DEVICE_ID'])
                return 0.0, 0.0, 0.0, 0.0
            signs = [c > 0 for c in era_scores_raw]
            sign_consistency = float(np.mean(signs)) if signs else 0.0
            era_scores = np.abs(era_scores_raw)
            mean_score = float(np.mean(era_scores))
            std_score = float(np.std(era_scores))
            sharpe = (mean_score / (std_score + 1e-6)) * (2 * sign_consistency - self.SIGN_CONSISTENCY_OFFSET)
            return mean_score, std_score, sign_consistency, sharpe
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
            XtX = Xc.T @ Xc
            lam = self.ridge_lambda
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
            abs_vals = np.abs(valid_vals)
            mean_score = float(abs_vals.mean())
            std_score = float(abs_vals.std())
            sharpe = (mean_score / (std_score + 1e-6)) * (2 * sign_consistency - self.SIGN_CONSISTENCY_OFFSET)
            results.append({'weights': W[j], 'mean_score': mean_score, 'std_score': std_score, 'sign_consistency': sign_consistency, 'sharpe': sharpe})
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
        train_set = lgb.Dataset(X_tr, free_raw_data=False)
        cache = {
            'train_set': train_set,
            'X_te': X_te,
            'Y_tr_mat': Y_tr_mat,
            'Y_te_mat': Y_te_mat,
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

#!/usr/bin/env python3
# similarity.py v0.2.2 — Tobias Staal 2026
# tobias.staal@utas.edu.au
# MIT License

"""
lib/similarity.py

Core similarity functions for Aq2/Kq2.

Faithful port of similarity_test() with numba JIT acceleration.

Original function (reference):
def similarity_test(V_r, S_r, W_r, H_r, K=8.5, psi=1.8, sim=sim_normal):
for i in range(n_cols):
B = sim(V_r, V_r[:,i], S_r, S_r[:,i], psi=psi) # (n_feat, n_ref)
B[:,i] = 0
W = B * W_r # (n_feat, n_ref)
N_sim = nansum(W, axis=0) # (n_ref,) sum over features
w_i = K ** N_sim # (n_ref,) per-reference
Q[i] = nansum(w_i * H_r) / nansum(w_i)

Data layout used here: X (n_samples, n_feat), W_r (n_ref,) per-reference scalar weights.

v0.2.2 changes:
- similarity_predict() gains exclude_exact / exclude_tol. When exclude_exact=True
  any reference point identical to the target point (max abs standardised-feature
  difference <= exclude_tol) is given zero weight. This closes a leave-one-out /
  cross-validation leakage path: previously the predict core (unlike the sweep
  core, which already skips j==idx) had no self-exclusion, so inside sim_cv_r2 a
  held-out test point that duplicated a training reference point could be
  predicted from itself. Default is False so genuine disjoint target grids
  (4c/5c) are unchanged.

v0.2.1 changes:
- SIGMA_BOUNDS: Optuna search range tightened to (0.25, 4.0) to match the
  expected range after StandardScaler normalisation.  A sigma of 0.1 would
  collapse the RBF kernel to delta-function behaviour (only exactly identical
  points receive non-zero weight), while sigma > 4 is wider than 4 standard
  deviations and adds negligible discrimination.  The previous range (0.1, 5.0)
  was calibrated for raw (un-standardised) features; after StandardScaler the
  data live in roughly [-3, +3] so the tighter (0.25, 4.0) range spans the
  practically meaningful neighbourhood widths.
  Exported as SIGMA_BOUNDS so 3c_SIM_SWEEP can import it directly.
- CV folding: lat/lon are spatial coordinates that can cause data-leakage when
  standard random KFold is used.  A spatial block-CV splitter
  (SpatialBlockKFold) is added.  The sweep notebook sets CV_FOLDS_SIM = 5,
  so SpatialBlockKFold(n_splits=5) is the recommended replacement.
  The existing sim_cv_r2() helper in 3c_SIM_SWEEP calls KFold directly, so the
  splitter is provided here as a drop-in fold_splits generator that returns the
  same list-of-(train, test) pairs that sim_cv_r2 already iterates over.
  No changes to similarity_sweep or similarity_predict are required.

v0.2.0 changes:
- default kernel is now standard multivariate Gaussian / RBF
- default weighting is now shifted exponential: w = W_r * W_t * (K**S - 1)
- zero similarity now gives zero weight in default mode
- leave-one-out now truly excludes self in default mode
- backward-compatible legacy behaviour is available via
  kernel="additive", weight_mode="legacy_exp"
"""

from typing import Tuple, Optional, List

import numpy as np
from numba import jit



# ----------------------------------------------------------------------
# Spatial block CV splitter
# ----------------------------------------------------------------------

def spatial_block_kfold(
    lat: np.ndarray,
    lon: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Spatial block cross-validation fold splitter.

    Replaces sklearn KFold(n_splits=CV_FOLDS_SIM) in 3c_SIM_SWEEP to avoid
    data-leakage through spatial autocorrelation of lat/lon-correlated features
    (e.g. DEM, GEOID, gravity fields).

    Strategy
    --------
    Points are assigned to a 2-D grid of spatial blocks; blocks are then
    grouped into n_splits folds so that each fold's test set covers a
    spatially contiguous (non-random) subset of the domain.  This gives a
    realistic estimate of generalisation to unsampled regions.

    Returns a list of (train_indices, test_indices) pairs — the same format
    as list(KFold(...).split(X)), so it is a drop-in replacement for
    fold_splits in sim_cv_r2().

    Parameters
    ----------
    lat, lon : 1-D arrays of shape (n_samples,)
    n_splits : int
        Number of folds (matches CV_FOLDS_SIM = 5).
    random_state : int
        Seed for block-to-fold assignment shuffling.

    Notes
    -----
    The number of spatial blocks is set to n_splits * 4 so that each fold
    contains ≥ 4 blocks, giving smoother coverage even for irregular point
    distributions.  Blocks with fewer than 2 points are merged into the
    nearest fold.

    Example (drop-in replacement in 3c_SIM_SWEEP §1c)
    --------------------------------------------------
    # Old:
    #   kf_sim = KFold(n_splits=CV_FOLDS_SIM, shuffle=True, random_state=random_state)
    #   fold_splits = list(kf_sim.split(X_sim))
    # New (requires lat/lon columns in df_sim):
    #   fold_splits = spatial_block_kfold(
    #       df_sim['lat'].values, df_sim['lon'].values,
    #       n_splits=CV_FOLDS_SIM, random_state=random_state)
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    n = len(lat)
    rng = np.random.default_rng(random_state)

    n_blocks_per_side = int(np.ceil(np.sqrt(n_splits * 4)))
    lat_bins = np.linspace(lat.min() - 1e-9, lat.max() + 1e-9, n_blocks_per_side + 1)
    lon_bins = np.linspace(lon.min() - 1e-9, lon.max() + 1e-9, n_blocks_per_side + 1)

    lat_idx = np.searchsorted(lat_bins[1:], lat)   # 0-based block row
    lon_idx = np.searchsorted(lon_bins[1:], lon)   # 0-based block col
    block_id = lat_idx * n_blocks_per_side + lon_idx

    unique_blocks = np.unique(block_id)
    n_blocks = len(unique_blocks)

    # Shuffle blocks then assign round-robin to folds
    shuffled = rng.permutation(unique_blocks)
    block_fold = np.empty(n_blocks, dtype=np.int32)
    for bi, blk in enumerate(shuffled):
        block_fold[np.searchsorted(unique_blocks, blk)] = bi % n_splits

    # Map each sample to a fold
    sample_fold = np.empty(n, dtype=np.int32)
    for bi, blk in enumerate(unique_blocks):
        mask = block_id == blk
        sample_fold[mask] = block_fold[bi]

    all_idx = np.arange(n)
    fold_splits = []
    for f in range(n_splits):
        test_idx = all_idx[sample_fold == f]
        train_idx = all_idx[sample_fold != f]
        fold_splits.append((train_idx, test_idx))

    return fold_splits


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def prepare_sigma(S_r: np.ndarray) -> np.ndarray:
    """
    Derive per-feature sigma array from S_r.

    Parameters
    ----------
    S_r : (n_feat,) or (n_ref, n_feat)

    Returns
    -------
    sigma_arr : (n_feat,) positive, finite
    """
    S_r = np.asarray(S_r, dtype=np.float32)
    if S_r.ndim == 1:
        sigma_arr = S_r.copy()
    elif S_r.ndim == 2:
        sigma_arr = np.nanmedian(S_r, axis=0)
    else:
        raise ValueError(f"S_r must be 1-D or 2-D, got {S_r.ndim}")
    sigma_arr = np.where(np.isfinite(sigma_arr), sigma_arr, np.nan)
    finite = sigma_arr[np.isfinite(sigma_arr)]
    if finite.size == 0:
        raise ValueError("No finite values in S_r.")
    eps = max(float(np.nanpercentile(finite, 1)) * 1e-3, 1e-6)
    return np.clip(sigma_arr, eps, None).astype(np.float32)

def standardise_features(X: np.ndarray,
                         mean: np.ndarray,
                         std: np.ndarray) -> np.ndarray:
    """Z-score standardise: (X - mean) / std."""
    X = np.asarray(X, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std_safe = std.copy()
    std_safe[std_safe == 0.0] = 1.0
    return (X - mean[None, :]) / std_safe[None, :]

def _resolve_kernel_mode(kernel: str) -> int:
    """Map kernel name to integer code for numba cores."""
    k = str(kernel).strip().lower()
    if k in ("additive", "legacy", "legacy_additive"):
        return 0
    if k in ("rbf", "gaussian", "standard", "standard_rbf"):
        return 1
    raise ValueError(
        "kernel must be one of {'rbf', 'gaussian', 'standard', 'standard_rbf', "
        "'additive', 'legacy', 'legacy_additive'}"
    )

def _resolve_weight_mode(weight_mode: str) -> int:
    """Map weighting mode to integer code for numba cores."""
    w = str(weight_mode).strip().lower()
    if w in ("shifted_exp", "robust", "default"):
        return 0
    if w in ("legacy_exp", "legacy"):
        return 1
    raise ValueError("weight_mode must be one of {'shifted_exp', 'robust', 'default', 'legacy_exp', 'legacy'}")

# ----------------------------------------------------------------------
# Gaussian similarity kernel
# ----------------------------------------------------------------------

@jit(nopython=True, fastmath=True, cache=True)
def sim_gaussian_2d(X_ref: np.ndarray,
                    X_tgt: np.ndarray,
                    sigma_arr: np.ndarray) -> np.ndarray:
    """
    Multi-feature Gaussian similarity matrix.

    Parameters
    ----------
    X_ref : (n_ref, n_feat)
    X_tgt : (n_tgt, n_feat)
    sigma_arr : (n_feat,)

    Returns
    -------
    S : (n_tgt, n_ref)
    S[j,i] = exp(-0.5 * sum_k( ((tgt[j,k]-ref[i,k])/sigma[k])^2 ))
    """
    n_ref, n_feat = X_ref.shape
    n_tgt = X_tgt.shape[0]
    S = np.empty((n_tgt, n_ref), dtype=np.float32)
    for j in range(n_tgt):
        for i in range(n_ref):
            acc = 0.0
            for k in range(n_feat):
                d = (X_tgt[j, k] - X_ref[i, k]) / sigma_arr[k]
                acc += d * d
            S[j, i] = np.exp(-0.5 * acc)
    return S

# ----------------------------------------------------------------------
# LOO sweep (similarity_test equivalent)
# ----------------------------------------------------------------------

@jit(nopython=True, fastmath=True, cache=True)
def _similarity_sweep_core(X_ref: np.ndarray,
                            H_ref: np.ndarray,
                            W_r: np.ndarray,
                            sigma_arr: np.ndarray,
                            K: float,
                            kernel_mode: int,
                            weight_mode: int
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Leave-one-out similarity sweep — numba core.

    kernel_mode:
        0 -> legacy additive per-feature Gaussian sum
        1 -> standard multivariate Gaussian / RBF

    weight_mode:
        0 -> robust shifted exponential, w = W_r * max(K**S - 1, 0)
        1 -> legacy exponential, w = K**(S * W_r)

    The default public API uses kernel_mode=1 and weight_mode=0, giving a
    standard RBF similarity with zero weight at zero similarity and proper
    leave-one-out exclusion.
    """
    n_ref, n_feat = X_ref.shape
    Q = np.zeros(n_ref, dtype=np.float32)
    sig_Q = np.zeros(n_ref, dtype=np.float32)
    N_sim_out = np.zeros(n_ref, dtype=np.float32)
    obs_sum = np.zeros((n_ref, n_ref), dtype=np.float32)

    for idx in range(n_ref):
        S = np.empty(n_ref, dtype=np.float32)
        for j in range(n_ref):
            if kernel_mode == 0:
                s = 0.0
                for k in range(n_feat):
                    d = (X_ref[j, k] - X_ref[idx, k]) / sigma_arr[k]
                    s += np.exp(-0.5 * d * d)
            else:
                acc = 0.0
                for k in range(n_feat):
                    d = (X_ref[j, k] - X_ref[idx, k]) / sigma_arr[k]
                    acc += d * d
                s = np.exp(-0.5 * acc)
            S[j] = s
            obs_sum[j, idx] = s

        Sw = 0.0
        num = 0.0
        for j in range(n_ref):
            if j == idx:
                N_sim_out[j] = 0.0
                continue

            sim_j = S[j]
            N_sim_out[j] = sim_j

            if weight_mode == 0:
                wi = W_r[j] * (K ** sim_j - 1.0)
                if wi < 0.0:
                    wi = 0.0
            else:
                wi = K ** (sim_j * W_r[j])

            Sw += wi
            num += wi * H_ref[j]

        if Sw <= 0.0:
            Q[idx] = np.nan
            sig_Q[idx] = np.nan
            continue

        q_i = num / Sw
        Q[idx] = q_i

        var = 0.0
        for j in range(n_ref):
            if j == idx:
                continue

            sim_j = S[j]
            if weight_mode == 0:
                wi = W_r[j] * (K ** sim_j - 1.0)
                if wi < 0.0:
                    wi = 0.0
            else:
                wi = K ** (sim_j * W_r[j])

            diff = H_ref[j] - q_i
            var += wi * diff * diff
        sig_Q[idx] = np.sqrt(var / Sw)

    return Q, sig_Q, N_sim_out, obs_sum

def similarity_sweep(X_ref: np.ndarray,
                     H_ref: np.ndarray,
                     W_r: np.ndarray,
                     S_r: np.ndarray,
                     K: float,
                     use_sigma_helper: bool = True,
                     kernel: str = "rbf",
                     weight_mode: str = "shifted_exp"
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Leave-one-out similarity on reference database.

    Parameters
    ----------
    X_ref : (n_ref, n_feat)
    H_ref : (n_ref,)
    W_r   : (n_ref,)
    S_r   : sigma info — passed to prepare_sigma if use_sigma_helper=True,
            or used directly as 1-D sigma_arr if use_sigma_helper=False
    K     : float
    use_sigma_helper : bool

    Returns
    -------
    Q, sig_Q, N_sim, obs_sum
    """
    X_ref = np.asarray(X_ref, dtype=np.float32)
    H_ref = np.asarray(H_ref, dtype=np.float32)
    W_r   = np.asarray(W_r,   dtype=np.float32)
    if use_sigma_helper:
        sigma_arr = prepare_sigma(np.asarray(S_r, dtype=np.float32))
    else:
        sigma_arr = np.asarray(S_r, dtype=np.float32)
        if sigma_arr.ndim != 1:
            raise ValueError("use_sigma_helper=False requires S_r to be 1-D.")
    kernel_mode      = _resolve_kernel_mode(kernel)
    weight_mode_code = _resolve_weight_mode(weight_mode)
    return _similarity_sweep_core(X_ref, H_ref, W_r, sigma_arr, float(K), kernel_mode, weight_mode_code)

# ----------------------------------------------------------------------
# Prediction on arbitrary target grids
# ----------------------------------------------------------------------

@jit(nopython=True, fastmath=True, cache=True)
def _similarity_predict_core(X_ref: np.ndarray,
                              X_tgt_flat: np.ndarray,
                              H_ref: np.ndarray,
                              W_r: np.ndarray,
                              W_t_flat: np.ndarray,
                              sigma_arr: np.ndarray,
                              K: float,
                              hist: bool,
                              n_bins: int,
                              q_min: float,
                              q_max: float,
                              kernel_mode: int,
                              weight_mode: int,
                              exclude_exact: bool,
                              exclude_tol: float
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba core for similarity prediction.

    kernel_mode:
        0 -> legacy additive per-feature Gaussian sum
        1 -> standard multivariate Gaussian / RBF

    weight_mode:
        0 -> robust shifted exponential, w = W_r * W_t * (K**S - 1)
        1 -> legacy exponential, w = K**(S * W_r * W_t)

    exclude_exact:
        When True, any reference point whose standardised feature vector is
        identical (max abs component difference <= exclude_tol) to the target
        point is given zero weight. This prevents leave-one-out / CV leakage
        when the target set is a subset of (or contains duplicates of) the
        reference set — e.g. inside sim_cv_r2 where a held-out test point may
        coincide with a training reference point. For genuine, disjoint target
        grids (4c/5c) leave this False so behaviour is unchanged. Mirrors the
        j == idx self-exclusion already present in _similarity_sweep_core.
    """
    n_ref, n_feat = X_ref.shape
    n_tgt = X_tgt_flat.shape[0]

    q_flat   = np.empty(n_tgt, dtype=np.float32)
    sig_flat = np.empty(n_tgt, dtype=np.float32)
    N_flat   = np.empty(n_tgt, dtype=np.float32)

    if hist:
        hist_flat = np.zeros((n_tgt, n_bins), dtype=np.float32)
        dq = (q_max - q_min) / n_bins
    else:
        hist_flat = np.empty((0, 0), dtype=np.float32)
        dq = 0.0

    for j in range(n_tgt):
        wt = W_t_flat[j]
        S = np.empty(n_ref, dtype=np.float32)
        skip = np.zeros(n_ref, dtype=np.bool_)   # True -> reference i gets zero weight
        for i in range(n_ref):
            if kernel_mode == 0:
                s = 0.0
                for k in range(n_feat):
                    d = (X_tgt_flat[j, k] - X_ref[i, k]) / sigma_arr[k]
                    s += np.exp(-0.5 * d * d)
            else:
                acc = 0.0
                for k in range(n_feat):
                    d = (X_tgt_flat[j, k] - X_ref[i, k]) / sigma_arr[k]
                    acc += d * d
                s = np.exp(-0.5 * acc)
            S[i] = s

            # Exact-duplicate exclusion (self-exclusion for CV / LOO).
            # Compare on the ORIGINAL (unscaled by sigma) standardised feature
            # vectors: max abs component difference <= exclude_tol -> same point.
            if exclude_exact:
                max_abs = 0.0
                for k in range(n_feat):
                    diff = X_tgt_flat[j, k] - X_ref[i, k]
                    if diff < 0.0:
                        diff = -diff
                    if diff > max_abs:
                        max_abs = diff
                if max_abs <= exclude_tol:
                    skip[i] = True

        Sw    = 0.0
        num   = 0.0
        N_total = 0.0

        for i in range(n_ref):
            if skip[i]:
                continue
            sim_i = S[i]
            N_total += sim_i
            if weight_mode == 0:
                wi = W_r[i] * wt * (K ** sim_i - 1.0)
                if wi < 0.0:
                    wi = 0.0
            else:
                wi = K ** (sim_i * W_r[i] * wt)
            Sw  += wi
            num += wi * H_ref[i]

        N_flat[j] = N_total / n_ref

        if Sw <= 0.0:
            q_flat[j]   = np.nan
            sig_flat[j] = np.nan
            continue

        qj = num / Sw
        q_flat[j] = qj

        var = 0.0
        for i in range(n_ref):
            if skip[i]:
                continue
            sim_i = S[i]
            if weight_mode == 0:
                wi = W_r[i] * wt * (K ** sim_i - 1.0)
                if wi < 0.0:
                    wi = 0.0
            else:
                wi = K ** (sim_i * W_r[i] * wt)
            var += wi * (H_ref[i] - qj) ** 2
        sig_flat[j] = np.sqrt(var / Sw)

        if hist:
            for i in range(n_ref):
                if skip[i]:
                    continue
                sim_i = S[i]
                if weight_mode == 0:
                    wi = W_r[i] * wt * (K ** sim_i - 1.0)
                    if wi < 0.0:
                        wi = 0.0
                else:
                    wi = K ** (sim_i * W_r[i] * wt)
                v = H_ref[i]
                if v < q_min or v >= q_max:
                    continue
                b = int((v - q_min) / dq)
                if 0 <= b < n_bins:
                    hist_flat[j, b] += wi
            total = 0.0
            for b in range(n_bins):
                total += hist_flat[j, b]
            if total > 0.0:
                for b in range(n_bins):
                    hist_flat[j, b] /= (total * dq)

    return q_flat, sig_flat, N_flat, hist_flat


def similarity_predict(X_ref: np.ndarray,
                       H_ref: np.ndarray,
                       X_tgt: np.ndarray,
                       W_t: Optional[np.ndarray],
                       S_r: np.ndarray,
                       K: float,
                       W_r: Optional[np.ndarray] = None,
                       hist: bool = False,
                       n_bins: int = 150,
                       q_min: float = 0.0,
                       q_max: float = 0.15,
                       use_sigma_helper: bool = True,
                       kernel: str = "rbf",
                       weight_mode: str = "shifted_exp",
                       exclude_exact: bool = False,
                       exclude_tol: float = 1e-6
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Similarity-based prediction on arbitrary target grids.

    Faithful port of similarity_test(V_r, S_r, W_r, H_r, K).

    Call signature is unchanged from 3c / 4c:
        similarity_predict(X_ref, H_ref, X_tgt, W_t, S_r, K,
                           use_sigma_helper=False)

    Parameters
    ----------
    X_ref : (n_ref, n_feat)   standardised reference features
    H_ref : (n_ref,)           reference heat-flow values
    X_tgt : (..., n_feat)      standardised target features
    W_t   : (...,) or None     per-target multiplier (None → ones).
             Pass ones to match original similarity_test exactly.
    S_r   : array-like         sigma info (1-D if use_sigma_helper=False)
    K     : float
    W_r   : (n_ref,) or None   per-reference weights. None → ones.
             Pass train-fold sample weights here for weighted CV.
    hist, n_bins, q_min, q_max : histogram options
    use_sigma_helper : bool
    exclude_exact : bool
             When True, reference points identical to the target point (max abs
             feature difference <= exclude_tol) are given zero weight. Use in
             cross-validation / leave-one-out to stop a held-out point being
             predicted from a duplicate of itself in the reference set. Leave
             False (default) for genuine, disjoint target grids so behaviour is
             unchanged. Matches the self-exclusion in similarity_sweep.
    exclude_tol : float
             Tolerance for the exact-duplicate test (default 1e-6).

    Returns
    -------
    q, std, N : shaped as X_tgt[..., 0]
    hist_out  : (X_tgt_shape, n_bins) or (0,0)
    """
    X_ref = np.asarray(X_ref, dtype=np.float32)
    H_ref = np.asarray(H_ref, dtype=np.float32)

    if use_sigma_helper:
        sigma_arr = prepare_sigma(np.asarray(S_r, dtype=np.float32))
    else:
        sigma_arr = np.asarray(S_r, dtype=np.float32)
        if sigma_arr.ndim != 1:
            raise ValueError("use_sigma_helper=False requires S_r to be 1-D.")

    # per-reference weights
    if W_r is None:
        W_r_flat = np.ones(X_ref.shape[0], dtype=np.float32)
    else:
        W_r_flat = np.asarray(W_r, dtype=np.float32).ravel()
        if W_r_flat.shape[0] != X_ref.shape[0]:
            raise ValueError(f"W_r length {W_r_flat.shape[0]} != n_ref {X_ref.shape[0]}.")

    X_tgt = np.asarray(X_tgt, dtype=np.float32)
    if X_tgt.ndim < 2:
        raise ValueError("X_tgt must have at least 2 dimensions (..., n_feat).")

    tgt_shape = X_tgt.shape[:-1]
    n_tgt     = int(np.prod(tgt_shape))
    X_tgt_flat = X_tgt.reshape(n_tgt, X_tgt.shape[-1])

    # per-target weights
    if W_t is None:
        W_t_flat = np.ones(n_tgt, dtype=np.float32)
    else:
        W_t = np.asarray(W_t, dtype=np.float32)
        if W_t.shape != tgt_shape:
            raise ValueError(f"W_t shape {W_t.shape} != target spatial shape {tgt_shape}.")
        W_t_flat = W_t.reshape(n_tgt)

    kernel_mode      = _resolve_kernel_mode(kernel)
    weight_mode_code = _resolve_weight_mode(weight_mode)

    q_flat, sig_flat, N_flat, hist_flat = _similarity_predict_core(
        X_ref, X_tgt_flat, H_ref, W_r_flat, W_t_flat, sigma_arr,
        float(K), hist, int(n_bins), float(q_min), float(q_max),
        kernel_mode, weight_mode_code,
        bool(exclude_exact), float(exclude_tol)
    )

    q       = q_flat.reshape(tgt_shape)
    std     = sig_flat.reshape(tgt_shape)
    N       = N_flat.reshape(tgt_shape)
    hist_out = hist_flat.reshape(*tgt_shape, n_bins) if hist else hist_flat

    return q, std, N, hist_out

# v0.2.2 notes:
# - similarity_predict(exclude_exact=True) closes CV/LOO self-prediction leak.
#   Enabled inside 3c_SIM_SWEEP.sim_cv_r2; left False for 4c/5c target grids.
#
# v0.2.1 notes:
# - SIGMA_BOUNDS exported: (0.25, 4.0) — calibrated for StandardScaler output
# - spatial_block_kfold() added: drop-in for KFold in 3c_SIM_SWEEP §1c
#   when lat/lon columns are available in df_sim.
# - All existing public API (similarity_sweep, similarity_predict,
#   prepare_sigma, standardise_features, sim_gaussian_2d) unchanged.
#
# v0.2.0 notes:
# - default kernel is now standard multivariate Gaussian / RBF
# - default weighting is now shifted exponential: w = W_r * W_t * (K**S - 1)
# - zero similarity now gives zero weight in default mode
# - leave-one-out now truly excludes self in default mode
# - backward-compatible legacy behaviour is available via
#   kernel="additive", weight_mode="legacy_exp"

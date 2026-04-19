#!/usr/bin/env python3
# similarity.py v0.1.3 — Tobias Staal 2026
# tobias.staal@utas.edu.au
# MIT License

"""
lib/similarity.py

Core similarity functions for Aq2/Kq2.

Faithful port of similarity_test() with numba JIT acceleration.

Original function (reference):
    def similarity_test(V_r, S_r, W_r, H_r, K=8.5, psi=1.8, sim=sim_normal):
        for i in range(n_cols):
            B = sim(V_r, V_r[:,i], S_r, S_r[:,i], psi=psi)  # (n_feat, n_ref)
            B[:,i] = 0
            W = B * W_r                                       # (n_feat, n_ref)
            N_sim = nansum(W, axis=0)                         # (n_ref,) sum over features
            w_i   = K ** N_sim                                # (n_ref,) per-reference
            Q[i]  = nansum(w_i * H_r) / nansum(w_i)

Data layout used here: X (n_samples, n_feat), W_r (n_ref,) per-reference scalar weights.
N_sim[j] = W_r[j] * sum_k( B(tgt, ref_j, k) )  — sum of per-feature similarities
           scaled by the reference weight. This is the per-reference analogue of the
           original nansum(B * W_r, axis=0).

K is meaningful because N_sim varies per reference, so w_i = K**N_sim[j] varies too.

v0.1.2 fix: _similarity_predict_core previously computed N_sim as a scalar
(sum over references), making w_scalar = K**N_sim a global cancel-out constant.
Now N_sim is computed per-reference (sum over features * W_r[j]), matching the
original, so K genuinely sharpens the neighbourhood.
"""

from typing import Tuple, Optional

import numpy as np
from numba import jit


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
    sigma_arr : (n_feat,)  positive, finite
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
    X    = np.asarray(X,    dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std  = np.asarray(std,  dtype=np.float32)
    std_safe = std.copy()
    std_safe[std_safe == 0.0] = 1.0
    return (X - mean[None, :]) / std_safe[None, :]


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
    X_ref     : (n_ref, n_feat)
    X_tgt     : (n_tgt, n_feat)
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
# LOO sweep  (similarity_test equivalent)
# ----------------------------------------------------------------------

@jit(nopython=True, fastmath=True, cache=True)
def _similarity_sweep_core(X_ref: np.ndarray,
                            H_ref: np.ndarray,
                            W_r: np.ndarray,
                            sigma_arr: np.ndarray,
                            K: float
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Leave-one-out similarity sweep — numba core.

    For test point idx:
        B[j]     = sum_k exp(-0.5*((X_ref[j,k]-X_ref[idx,k])/sigma[k])^2)
        B[idx]   = 0  (remove self)
        N_sim[j] = B[j] * W_r[j]          # per-reference, sum over features * weight
        w_i[j]   = K ** N_sim[j]           # per-reference — K is meaningful
        Q[idx]   = sum(w_i * H_r) / sum(w_i)

    Parameters
    ----------
    X_ref     : (n_ref, n_feat)
    H_ref     : (n_ref,)
    W_r       : (n_ref,)  per-reference scalar weights
    sigma_arr : (n_feat,)
    K         : float

    Returns
    -------
    Q, sig_Q  : (n_ref,)
    N_sim_last: (n_ref,)  N_sim from last iteration (diagnostics)
    obs_sum   : (n_ref, n_ref)  B matrix across all test points (diagnostics)
    """
    n_ref, n_feat = X_ref.shape
    Q         = np.zeros(n_ref, dtype=np.float32)
    sig_Q     = np.zeros(n_ref, dtype=np.float32)
    N_sim_out = np.zeros(n_ref, dtype=np.float32)
    obs_sum   = np.zeros((n_ref, n_ref), dtype=np.float32)

    for idx in range(n_ref):
        # B[j] = sum of per-feature Gaussian similarities
        B = np.empty(n_ref, dtype=np.float32)
        for j in range(n_ref):
            s = 0.0
            for k in range(n_feat):
                d = (X_ref[j, k] - X_ref[idx, k]) / sigma_arr[k]
                s += np.exp(-0.5 * d * d)
            B[j] = s
            obs_sum[j, idx] = s
        B[idx] = 0.0  # leave-one-out

        # N_sim[j] = B[j] * W_r[j]  — per-reference
        # w_i[j]   = K ** N_sim[j]  — varies per reference
        Sw  = 0.0
        num = 0.0
        for j in range(n_ref):
            N_sim_j = B[j] * W_r[j]
            N_sim_out[j] = N_sim_j
            wi   = K ** N_sim_j
            Sw  += wi
            num += wi * H_ref[j]

        if Sw <= 0.0:
            Q[idx]     = np.nan
            sig_Q[idx] = np.nan
            continue

        q_i        = num / Sw
        Q[idx]     = q_i

        var = 0.0
        for j in range(n_ref):
            N_sim_j = B[j] * W_r[j]
            wi       = K ** N_sim_j
            diff     = H_ref[j] - q_i
            var     += wi * diff * diff
        sig_Q[idx] = np.sqrt(var / Sw)

    return Q, sig_Q, N_sim_out, obs_sum


def similarity_sweep(X_ref: np.ndarray,
                     H_ref: np.ndarray,
                     W_r: np.ndarray,
                     S_r: np.ndarray,
                     K: float,
                     use_sigma_helper: bool = True
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
    return _similarity_sweep_core(X_ref, H_ref, W_r, sigma_arr, float(K))


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
                              q_max: float
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba core for similarity prediction. Faithful port of similarity_test().

    For each target point j:
        B[i]     = sum_k exp(-0.5*((X_tgt[j,k]-X_ref[i,k])/sigma[k])^2)
                   (sum of per-feature Gaussian similarities to reference i)
        N_sim[i] = B[i] * W_r[i]   — per-reference, varies across i
        w_i[i]   = K ** N_sim[i]   — per-reference, K is meaningful
        q[j]     = sum_i(w_i * H_ref[i]) / sum_i(w_i)

    W_t_flat is an optional per-target scalar multiplier on B (kept for
    API compatibility with 3c sim_cv_r2 which passes w_te / ones).
    It does NOT affect K sensitivity because it scales B uniformly for
    all references at a given target, and therefore scales N_sim uniformly —
    but since w_i = K**N_sim[i] and K**( c*x ) != c * K**x, W_t does have
    a non-trivial effect on the sharpening. Pass ones to match the original
    similarity_test exactly.

    Parameters
    ----------
    X_ref      : (n_ref, n_feat)
    X_tgt_flat : (n_tgt, n_feat)
    H_ref      : (n_ref,)
    W_r        : (n_ref,)   per-reference weights  ← KEY: makes N_sim per-reference
    W_t_flat   : (n_tgt,)   per-target multiplier (pass ones for exact original)
    sigma_arr  : (n_feat,)
    K          : float
    hist       : bool
    n_bins     : int
    q_min, q_max : float

    Returns
    -------
    q_flat, sig_flat, N_flat, hist_flat
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
        wt = W_t_flat[j]  # per-target scalar (1.0 for original behaviour)

        # B[i] = sum of per-feature Gaussian sims between target j and reference i
        B = np.empty(n_ref, dtype=np.float32)
        for i in range(n_ref):
            s = 0.0
            for k in range(n_feat):
                d = (X_tgt_flat[j, k] - X_ref[i, k]) / sigma_arr[k]
                s += np.exp(-0.5 * d * d)
            B[i] = s

        # N_sim[i] = B[i] * W_r[i] * wt  — per-reference
        # w_i[i]   = K ** N_sim[i]        — per-reference, varies across i
        Sw      = 0.0
        num     = 0.0
        N_total = 0.0

        for i in range(n_ref):
            N_sim_i = B[i] * W_r[i] * wt
            wi       = K ** N_sim_i
            Sw      += wi
            num     += wi * H_ref[i]
            N_total += N_sim_i

        N_flat[j] = N_total / n_ref  # mean N_sim (diagnostics)

        if Sw <= 0.0:
            q_flat[j]   = np.nan
            sig_flat[j] = np.nan
            continue

        qj          = num / Sw
        q_flat[j]   = qj

        var = 0.0
        for i in range(n_ref):
            N_sim_i = B[i] * W_r[i] * wt
            wi       = K ** N_sim_i
            var     += wi * (H_ref[i] - qj) ** 2
        sig_flat[j] = np.sqrt(var / Sw)

        if hist:
            for i in range(n_ref):
                N_sim_i = B[i] * W_r[i] * wt
                wi = K ** N_sim_i
                v  = H_ref[i]
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
                       use_sigma_helper: bool = True
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
    H_ref : (n_ref,)          reference heat-flow values
    X_tgt : (..., n_feat)     standardised target features
    W_t   : (...,) or None    per-target multiplier (None → ones).
                              Pass ones to match original similarity_test exactly.
    S_r   : array-like        sigma info (1-D if use_sigma_helper=False)
    K     : float
    W_r   : (n_ref,) or None  per-reference weights. None → ones.
                              Pass train-fold sample weights here for weighted CV.
    hist, n_bins, q_min, q_max : histogram options
    use_sigma_helper : bool

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

    tgt_shape  = X_tgt.shape[:-1]
    n_tgt      = int(np.prod(tgt_shape))
    X_tgt_flat = X_tgt.reshape(n_tgt, X_tgt.shape[-1])

    # per-target weights
    if W_t is None:
        W_t_flat = np.ones(n_tgt, dtype=np.float32)
    else:
        W_t = np.asarray(W_t, dtype=np.float32)
        if W_t.shape != tgt_shape:
            raise ValueError(f"W_t shape {W_t.shape} != target spatial shape {tgt_shape}.")
        W_t_flat = W_t.reshape(n_tgt)

    q_flat, sig_flat, N_flat, hist_flat = _similarity_predict_core(
        X_ref, X_tgt_flat, H_ref, W_r_flat, W_t_flat, sigma_arr,
        float(K), hist, int(n_bins), float(q_min), float(q_max)
    )

    q        = q_flat.reshape(tgt_shape)
    std      = sig_flat.reshape(tgt_shape)
    N        = N_flat.reshape(tgt_shape)
    hist_out = hist_flat.reshape(*tgt_shape, n_bins) if hist else hist_flat

    return q, std, N, hist_out

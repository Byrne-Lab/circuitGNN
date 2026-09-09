"""Automatic bin-size selection via signed-edge stability (``signkappa_interior``).

``downsample_factor='auto'`` picks the bin whose signed lag-1 VAR coupling pattern is
most reproducible across time segments (mean pairwise Cohen's kappa of {+1,0,-1} edges).
The pick is the argmax of that curve, unless it climbs monotonically to the coarsest
candidate (no timescale resolved), in which case it falls back to ``FALLBACK_BIN_MS``.
"""
import numpy as np

CANDIDATE_BINS_MS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]  # bins the selector scores
FALLBACK_BIN_MS = 4.0   # used when the kappa curve has no interior peak
N_PROXY = 150           # max neurons selected randomly
RIDGE = 1.0             # ridge on every lag-1 VAR solve
SIGN_Z = 1.0            # threshold (std units) for calling an edge +/-/0
SEG_K = 4               # disjoint contiguous segments compared pairwise
SEED = 0                # selected bin is deterministic per recording


def _rebin(native, native_ms, bin_ms):
    """Sum native-resolution spikes into ``bin_ms`` bins."""
    f = max(1, int(round(bin_ms / native_ms)))
    n, T = native.shape
    trim = T // f * f # largest length <= T that is a multiple of f
    return native[:, :trim].reshape(n, -1, f).sum(-1)


def _subset(X, rng):
    """Random neuron subset (<= N_PROXY) as float."""
    if X.shape[0] > N_PROXY:
        idx = np.sort(rng.choice(X.shape[0], N_PROXY, replace=False))
        X = X[idx]
    return X.astype(float)


def _lag1_coupling(Xh): # Xh -> one contiguous time segment of the binned spike count matrix (N, T)
    """Ridge lag-1 VAR coupling matrix (N x N): entry [i,j] = neuron i at t-1 -> j at t."""
    A = Xh[:, :-1].T # predictor matrix (drops the last bin), each column is a neuron, row is population vector at time t-1
    B = Xh[:, 1:].T # target matrix (shifted forward by 1 bin)
    A = (A - A.mean(0)) / (A.std(0) + 1e-8) # z-score each column (neuron) across time
    N = Xh.shape[0]
    return np.linalg.solve(A.T @ A + RIDGE * np.eye(N), A.T @ B) 
    # A.T @ A is the NxN covariance of past states
    # A.T @ B is the NxN past->future cross-covariance


def _segment_couplings(X, K):
    """Off-diagonal lag-1 coupling vectors, one per contiguous segment."""
    N, Tb = X.shape
    off = ~np.eye(N, dtype=bool)
    return [_lag1_coupling(X[:, c])[off] for c in np.array_split(np.arange(Tb), K)]


def _cohen_kappa_3(a, b):
    """Cohen's kappa between two label vectors over categories {-1, 0, +1}."""
    po = float(np.mean(a == b)) # observed agreements
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in (-1, 0, 1))) # agreement expected by chance
    return (po - pe) / (1 - pe) if pe < 1.0 else np.nan 
    # num -> agreement above chance
    # den -> maximum possible agreement above chance


def sign_stability(spikes_native, native_ms, bin_ms, rng, K=SEG_K, z=SIGN_Z):
    """Mean pairwise Cohen's kappa of the per-segment signed edge labels at ``bin_ms``
    (each coupling thresholded to +1/0/-1 at |w| > z*std)."""
    X = _subset(_rebin(spikes_native, native_ms, bin_ms), rng)
    labs = []
    for v in _segment_couplings(X, K):
        thr = z * (v.std() + 1e-12)
        labs.append(np.where(v > thr, 1, np.where(v < -thr, -1, 0)))

    ks = [_cohen_kappa_3(labs[a], labs[b]) for a in range(len(labs)) for b in range(a + 1, len(labs))]
    ks = [k for k in ks if not np.isnan(k)]

    return float(np.mean(ks)) if ks else np.nan


def _interior_pick(curve, fallback_ms=FALLBACK_BIN_MS):
    """``(argmax(curve), True)`` unless the peak is the coarsest bin (monotone climb), in which case ``(fallback_ms, False)``. Ties -> finer bin."""
    good = {b: v for b, v in curve.items() if v is not None and not np.isnan(v)}
    if len(good) >= 3:
        peak = max(good, key=good.get)   # ties -> finer bin
        if peak != max(good):            # reject a monotone climb to the coarsest bin
            return peak, True
    return fallback_ms, False


def optimal_bin_ms(spikes_native, native_ms, seed=SEED, fallback_ms=FALLBACK_BIN_MS,
                   candidate_bins_ms=CANDIDATE_BINS_MS, verbose=True):
    """Return ``(bin_ms, curve)`` for a native-resolution recording, where ``curve`` maps
    each candidate bin to its signed-kappa stability."""
    curve = {b: sign_stability(spikes_native, native_ms, b, np.random.default_rng(seed)) for b in candidate_bins_ms}
    bin_ms, is_peak = _interior_pick(curve, fallback_ms)
    if verbose:
        how = "resolved peak" if is_peak else f"no peak -> fallback {fallback_ms:g} ms"
        print(f"[auto bin] Autobin -> bin={bin_ms:g} ms ({how})")
    return float(bin_ms), curve


def resolve_downsample(downsample_factor, spikes_native, native_ms,
                       seed=SEED, fallback_ms=FALLBACK_BIN_MS, verbose=True, **_):
    """Coerce ``downsample_factor`` to an int, resolving ``'auto'`` via the autobin pick."""
    if isinstance(downsample_factor, str):
        bin_ms, _ = optimal_bin_ms(spikes_native, native_ms, seed=seed, fallback_ms=fallback_ms, verbose=verbose)
        return max(1, int(round(bin_ms / native_ms)))
    return int(downsample_factor)

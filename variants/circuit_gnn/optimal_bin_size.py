"""Automatic bin-size selection via signed-edge stability (``signkappa_interior``).

``downsample_factor='auto'`` picks the bin whose SIGNED lag-1 coupling pattern is
most reproducible across disjoint time segments: each segment's ridge lag-1 VAR
coupling matrix is thresholded into {+1, 0, -1} edges and scored by the mean
pairwise Cohen's kappa. The chosen bin is the argmax of that curve **only if it is
an interior peak** (rises then turns down before the coarsest candidate); a
monotone climb to the boundary means no coupling timescale is resolved, so we fall
back to ``FALLBACK_BIN_MS``. This rule won the bakeoff in
``variants/results_bin_size/bin_search.py`` (the occupancy rule it replaced now
lives there as the ``occupancy(prod)`` baseline).
"""
import numpy as np

CANDIDATE_BINS_MS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]  # bins the selector scores
FALLBACK_BIN_MS = 4.0   # used when the kappa curve has no interior peak
N_PROXY = 150           # cap neurons (random subset) for speed + comparability
RIDGE = 1.0             # ridge on every lag-1 VAR solve
SIGN_Z = 1.0            # threshold (std units) for calling an edge +/-/0
SEG_K = 4               # disjoint contiguous segments compared pairwise
SEED = 0                # fixed -> the selected bin is deterministic per recording


def _rebin(native, native_ms, bin_ms):
    """Sum native-resolution spikes into ``bin_ms`` bins."""
    f = max(1, int(round(bin_ms / native_ms)))
    n, T = native.shape
    trim = T // f * f
    return native[:, :trim].reshape(n, -1, f).sum(-1) if trim else native.sum(1, keepdims=True)


def _subset(X, rng):
    """Random neuron subset (<= N_PROXY) as float, for speed and cross-dataset parity."""
    if X.shape[0] > N_PROXY:
        idx = np.sort(rng.choice(X.shape[0], N_PROXY, replace=False))
        X = X[idx]
    return X.astype(float)


def _lag1_coupling(Xh):
    """Ridge lag-1 VAR coupling matrix (N x N): entry [i,j] = neuron i at t-1 -> j at t."""
    A = Xh[:, :-1].T
    B = Xh[:, 1:].T
    A = (A - A.mean(0)) / (A.std(0) + 1e-8)
    N = Xh.shape[0]
    return np.linalg.solve(A.T @ A + RIDGE * np.eye(N), A.T @ B)


def _segment_couplings(X, K):
    """Off-diagonal lag-1 coupling vectors, one per contiguous segment (>30 bins)."""
    N, Tb = X.shape
    off = ~np.eye(N, dtype=bool)
    return [_lag1_coupling(X[:, c])[off]
            for c in np.array_split(np.arange(Tb), K) if c.size > 30]


def _cohen_kappa_3(a, b):
    """Cohen's kappa between two label vectors over categories {-1, 0, +1}."""
    if a.size == 0:
        return np.nan
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in (-1, 0, 1)))
    return (po - pe) / (1 - pe) if pe < 1.0 else np.nan


def sign_stability(spikes_native, native_ms, bin_ms, rng, K=SEG_K, z=SIGN_Z):
    """Reproducibility of the SIGNED edge pattern at ``bin_ms``. Threshold each segment's
    lag-1 coupling into {+1, 0, -1} (|w| > z*std -> its sign) and take the mean pairwise
    Cohen's kappa of the off-diagonal labels."""
    X = _subset(_rebin(spikes_native, native_ms, bin_ms), rng)
    N, Tb = X.shape
    if Tb < 40 * K or N < 3:
        return np.nan
    labs = []
    for v in _segment_couplings(X, K):
        thr = z * (v.std() + 1e-12)
        labs.append(np.where(v > thr, 1, np.where(v < -thr, -1, 0)))
    if len(labs) < 2:
        return np.nan
    ks = [_cohen_kappa_3(labs[a], labs[b])
          for a in range(len(labs)) for b in range(a + 1, len(labs))]
    ks = [k for k in ks if not np.isnan(k)]
    return float(np.mean(ks)) if ks else np.nan


def _interior_pick(curve, fallback_ms=FALLBACK_BIN_MS):
    """Argmax of ``curve`` if it is an INTERIOR peak (not at either boundary); else
    ``fallback_ms``. Ties break toward the finer bin (ascending candidate order)."""
    good = {b: v for b, v in curve.items() if v is not None and not np.isnan(v)}
    if len(good) < 3:
        return fallback_ms
    bins = sorted(good)
    peak = max(good, key=good.get)
    return peak if (peak != bins[0] and peak != bins[-1]) else fallback_ms


def optimal_bin_ms(spikes_native, native_ms, seed=SEED, fallback_ms=FALLBACK_BIN_MS,
                   candidate_bins_ms=CANDIDATE_BINS_MS, verbose=True):
    """signkappa_interior bin (ms) for a native-resolution recording.

    Returns ``(bin_ms, curve)`` where ``curve`` maps each candidate bin to its
    signed-kappa stability. A fresh RNG per candidate (same ``seed``) keeps the
    neuron subset identical across bins.
    """
    curve = {b: sign_stability(spikes_native, native_ms, b, np.random.default_rng(seed))
             for b in candidate_bins_ms}
    bin_ms = _interior_pick(curve, fallback_ms)
    interior = _interior_pick(curve, -1) != -1
    if verbose:
        how = "interior peak" if interior else f"no interior peak -> fallback {fallback_ms:g} ms"
        print(f"[auto bin] signkappa_interior -> bin={bin_ms:g} ms ({how})")
    return float(bin_ms), curve


def resolve_downsample(downsample_factor, spikes_native, native_ms,
                       seed=SEED, fallback_ms=FALLBACK_BIN_MS, verbose=True, **_):
    """Coerce a loader's ``downsample_factor`` to an int; resolves ``'auto'`` to the
    integer downsample factor whose bin width matches the signkappa_interior pick."""
    if isinstance(downsample_factor, str):
        if downsample_factor.lower() != "auto":
            raise ValueError(f"downsample_factor must be int or 'auto', got {downsample_factor!r}")
        bin_ms, _ = optimal_bin_ms(spikes_native, native_ms, seed=seed,
                                   fallback_ms=fallback_ms, verbose=verbose)
        return max(1, int(round(bin_ms / native_ms)))
    return int(downsample_factor)


# native bin width (ms) per dataset loader, for the standalone recommender
_NATIVE_MS = {"arbitrary": 1.0, "OC_2": 0.5, "pdcm": 1.0,
              "small_circuit_A": 0.5, "small_circuit_B": 0.5}


def recommend_for_dataset(name, **kwargs):
    """Load ``name`` at native resolution and report its signkappa_interior bin."""
    from circuit_gnn import data as _data  # lazy to avoid circular import
    if name not in _NATIVE_MS:
        raise KeyError(f"unknown dataset {name!r}; options: {list(_NATIVE_MS)}")
    spikes_native, _, _ = getattr(_data, f"load_{name}_network")(downsample_factor=1)
    return optimal_bin_ms(spikes_native, _NATIVE_MS[name], **kwargs)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # put src/ on path
    import matplotlib
    matplotlib.use("Agg")
    for name in ["arbitrary", "OC_2", "pdcm"]:
        print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
        bin_ms, curve = recommend_for_dataset(name)
        print(f"  bin={bin_ms:g} ms   kappa curve: "
              + "  ".join(f"{b}:{'-' if v is None or v != v else f'{v:.3f}'}"
                          for b, v in curve.items()))

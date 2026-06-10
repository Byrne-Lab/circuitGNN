import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import minimize
from scipy.interpolate import BSpline
from tqdm import tqdm


def _make_bspline_basis(n_bins, knot_spacing=5, degree=2):
    lags = np.arange(n_bins, dtype=float)
    interior_knots = np.arange(knot_spacing, n_bins, knot_spacing, dtype=float)
    knots = np.concatenate([
        np.zeros(degree + 1),
        interior_knots,
        np.full(degree + 1, float(n_bins)),
    ])
    n_basis = len(interior_knots) + degree + 1
    B = np.zeros((n_bins, n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        B[:, j] = BSpline(knots, c, degree)(lags)
    return B


def _build_features(spike_train, basis):
    T = len(spike_train)
    n_basis = basis.shape[1]
    features = np.zeros((T, n_basis))
    x = spike_train.astype(float)
    for j in range(n_basis):
        full = fftconvolve(x, basis[:, j])
        features[1:, j] = full[:T - 1]
    return features


def _fit_unit(binned_spikes, target_idx, self_basis, cross_basis, dt_ms, beta_min=-20.0):
    N, T = binned_spikes.shape
    n_self = self_basis.shape[1]
    n_cross = cross_basis.shape[1]
    others = [i for i in range(N) if i != target_idx]

    parts = [np.ones((T, 1)), _build_features(binned_spikes[target_idx], self_basis)]
    for nidx in others:
        parts.append(_build_features(binned_spikes[nidx], cross_basis))
    X = np.concatenate(parts, axis=1)
    y = binned_spikes[target_idx].astype(float)

    def nll_and_grad(beta):
        h = np.clip(X @ beta, -500.0, 500.0)
        p = np.clip(1.0 / (1.0 + np.exp(-h)), 1e-12, 1.0 - 1e-12)
        nll = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        grad = X.T @ (p - y)
        return nll, grad

    result = minimize(
        nll_and_grad,
        x0=np.zeros(X.shape[1]),
        jac=True,
        method='L-BFGS-B',
        bounds=[(beta_min, None)] * X.shape[1],
        options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6},
    )
    beta = result.x
    dt_s = dt_ms * 1e-3
    basis_integrals = cross_basis.sum(axis=0) * dt_s

    cs = {}
    for k, nidx in enumerate(others):
        b = beta[1 + n_self + k * n_cross: 1 + n_self + (k + 1) * n_cross]
        cs[nidx] = float(np.abs(b @ basis_integrals))
    return cs


def calculate_glm(binned_spikes, self_lag_ms=200, cross_lag_ms=100, knot_spacing_ms=5, degree=2, dt_ms=1.0):
    # Based on Gerhard et al. (2013)

    N = binned_spikes.shape[0]
    self_bins = max(1, int(self_lag_ms / dt_ms))
    cross_bins = max(1, int(cross_lag_ms / dt_ms))
    knot_bins = max(1, int(knot_spacing_ms / dt_ms))

    self_basis = _make_bspline_basis(self_bins, knot_spacing=knot_bins, degree=degree)
    cross_basis = _make_bspline_basis(cross_bins, knot_spacing=knot_bins, degree=degree)

    conn_matrix = np.zeros((N, N))
    for target_idx in tqdm(range(N), desc='GLM'):
        cs = _fit_unit(binned_spikes, target_idx, self_basis, cross_basis, dt_ms)
        for source_idx, val in cs.items():
            conn_matrix[source_idx, target_idx] = val
    return conn_matrix

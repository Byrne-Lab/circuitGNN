# Based on Gerhard et al. (2013)

import warnings

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import minimize
from scipy.interpolate import BSpline
from tqdm import tqdm

COEF_LOWER_BOUND = -20.0


def _make_bspline_basis(n_bins, knot_spacing=5, degree=2):
    # knot_spacing and degree set to match Gerhard et al. (2013)
    lags = np.arange(n_bins, dtype=float)  # time windows for history (n_bins lag bins)
    interior_knots = np.arange(knot_spacing, n_bins, knot_spacing, dtype=float)  # spacing of spline anchors
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
    return B  # (num_bins, num_basis)


def _build_features(spike_train, basis):
    T = len(spike_train)  # length of the spike train
    n_basis = basis.shape[1]
    features = np.zeros((T, n_basis))  # convolved spike histories
    x = spike_train.astype(float)
    for j in range(n_basis):
        full = fftconvolve(x, basis[:, j])
        features[1:, j] = full[:T - 1]
    return features  # feature matrix with T rows and n_basis columns


def _design_matrix(binned_spikes, target_idx, self_basis, cross_basis):
    # cross_basis=None builds a self-only model (used during self-lag selection).
    N, T = binned_spikes.shape
    others = [i for i in range(N) if i != target_idx]
    parts = [np.ones((T, 1)), _build_features(binned_spikes[target_idx], self_basis)]
    if cross_basis is not None:
        for nidx in others:
            parts.append(_build_features(binned_spikes[nidx], cross_basis))
    X = np.concatenate(parts, axis=1)
    return X, others


def _fit_design(X, y):
    # maximum-likelihood Bernoulli GLM fit
    n_params = X.shape[1]
    bounds = [(None, None)] + [(COEF_LOWER_BOUND, None)] * (n_params - 1)

    def nll_and_grad(beta):
        h = np.clip(X @ beta, -500.0, 500.0)
        p = np.clip(1.0 / (1.0 + np.exp(-h)), 1e-12, 1.0 - 1e-12)  # nonlinear transformation
        nll = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))  # negative log likelihood
        grad = X.T @ (p - y)
        return nll, grad

    result = minimize(
        nll_and_grad,
        x0=np.zeros(n_params),
        jac=True,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6},
    )
    return result.x, float(result.fun), n_params, len(y), bool(result.success)


def _bic(nll, n_params, n_samples):
    # BIC = -2 logL + (p/2) log(N)
    return 2.0 * nll + 0.5 * n_params * np.log(n_samples)


def _summed_bic(binned_spikes, self_basis, cross_basis):
    # BIC summed over all neurons of the data set
    total = 0.0
    for t in range(binned_spikes.shape[0]):
        X, _ = _design_matrix(binned_spikes, t, self_basis, cross_basis)
        _, nll, p, n, _ = _fit_design(X, binned_spikes[t].astype(float))
        total += _bic(nll, p, n)
    return total


def _select_lags(binned_spikes, dt_ms, knot_bins, degree, self_max_ms, cross_max_ms):
    # step the search by the knot spacing as the model only gains a basis function when the max lag crosses a knot
    knot_ms = knot_bins * dt_ms
    lo_ms = 3 * knot_ms  # at least a few basis functions

    # self-history lag, self-only models.
    self_grid = np.arange(lo_ms, self_max_ms + knot_ms, knot_ms)
    best_self, best_bic = self_grid[0], np.inf
    for lag in self_grid:
        sb = max(1, int(lag / dt_ms))
        sbasis = _make_bspline_basis(sb, knot_spacing=knot_bins, degree=degree)
        bic = _summed_bic(binned_spikes, sbasis, None)
        if bic < best_bic:
            best_bic, best_self = bic, lag

    self_basis = _make_bspline_basis(max(1, int(best_self / dt_ms)), knot_spacing=knot_bins, degree=degree)

    # cross-coupling lag, full models with the self lag fixed.
    cross_grid = np.arange(lo_ms, cross_max_ms + knot_ms, knot_ms)
    best_cross, best_bic = cross_grid[0], np.inf
    for lag in cross_grid:
        cb = max(1, int(lag / dt_ms))
        cbasis = _make_bspline_basis(cb, knot_spacing=knot_bins, degree=degree)
        bic = _summed_bic(binned_spikes, self_basis, cbasis)
        if bic < best_bic:
            best_bic, best_cross = bic, lag

    return float(best_self), float(best_cross)


def _fit_unit(binned_spikes, target_idx, self_basis, cross_basis, dt_ms):
    n_self = self_basis.shape[1]  # number of self-coupling basis functions
    n_cross = cross_basis.shape[1]  # number of cross-coupling basis functions
    X, others = _design_matrix(binned_spikes, target_idx, self_basis, cross_basis)
    y = binned_spikes[target_idx].astype(float)  # binary spike train of target neuron

    beta, _, _, _, success = _fit_design(X, y)
    if not success:
        warnings.warn(f"GLM fit for neuron {target_idx} did not converge to the requested tolerance")

    dt_s = dt_ms * 1e-3
    basis_integrals = cross_basis.sum(axis=0) * dt_s  # integral of each basis spline Dj(t)

    cs = {}
    for k, nidx in enumerate(others):
        b = beta[1 + n_self + k * n_cross: 1 + n_self + (k + 1) * n_cross]
        # signed coupling strength (net area under the cross-coupling filter):
        cs[nidx] = float(b @ basis_integrals)
    return cs


def calculate_glm(binned_spikes, self_lag_ms=None, cross_lag_ms=None, dt_ms=1.0, select_lags=True, self_max_ms=800.0, cross_max_ms=250.0):
    knot_spacing_ms = 5
    degree = 2

    N = binned_spikes.shape[0]
    knot_bins = max(1, int(knot_spacing_ms / dt_ms))

    if select_lags or self_lag_ms is None or cross_lag_ms is None:
        self_lag_ms, cross_lag_ms = _select_lags(
            binned_spikes, dt_ms, knot_bins, degree, self_max_ms, cross_max_ms)
        print(f"BIC-selected lags: self={self_lag_ms:.0f} ms, cross={cross_lag_ms:.0f} ms")

    self_bins = max(1, int(self_lag_ms / dt_ms))
    cross_bins = max(1, int(cross_lag_ms / dt_ms))

    self_basis = _make_bspline_basis(self_bins, knot_spacing=knot_bins, degree=degree)
    cross_basis = _make_bspline_basis(cross_bins, knot_spacing=knot_bins, degree=degree)

    conn_matrix = np.zeros((N, N))
    for target_idx in tqdm(range(N), desc='GLM'):
        cs = _fit_unit(binned_spikes, target_idx, self_basis, cross_basis, dt_ms)
        for source_idx, val in cs.items():
            conn_matrix[source_idx, target_idx] = val
    return conn_matrix
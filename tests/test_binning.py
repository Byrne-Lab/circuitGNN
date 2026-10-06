import numpy as np
import pytest

from circuit_gnn import optimal_bin_ms, rebin, spikes_from_times
from circuit_gnn.binning import CANDIDATE_BINS_MS, FALLBACK_BIN_MS


def test_rebin_sums_and_drops_the_remainder():
    x = np.arange(14).reshape(2, 7)
    assert rebin(x, 3).tolist() == [[0 + 1 + 2, 3 + 4 + 5], [7 + 8 + 9, 10 + 11 + 12]]


def test_spikes_from_times():
    trains = [np.array([0.0, 0.05, 0.11]), np.array([0.02, 0.09])]
    counts = spikes_from_times(trains, bin_ms=10.0, t_start=0.0, t_stop=0.12)
    assert counts.shape[0] == 2
    assert counts.sum() == 5


def test_spikes_from_times_rejects_bad_input():
    with pytest.raises(ValueError):
        spikes_from_times([np.array([0.1])], bin_ms=10.0)              # one neuron
    with pytest.raises(ValueError):
        spikes_from_times([np.array([]), np.array([])], bin_ms=10.0)   # all empty


def test_optimal_bin_ms_returns_a_candidate_or_the_fallback():
    spikes = np.random.default_rng(0).poisson(0.05, size=(8, 20000))
    bin_ms, curve = optimal_bin_ms(spikes, native_ms=1.0, verbose=False)
    assert list(curve) == CANDIDATE_BINS_MS
    assert bin_ms in CANDIDATE_BINS_MS or bin_ms == FALLBACK_BIN_MS

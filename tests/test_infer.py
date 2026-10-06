import numpy as np
import pytest

from circuit_gnn import Config, infer_connectivity, load_runs


def test_infer_connectivity_shapes_and_saved_runs(tmp_path):
    # ablation_threshold=-inf keeps every run: a 1-epoch model has no reason to pass
    # the real gate, and this test is about plumbing, not recovery quality.
    spikes = np.random.default_rng(0).poisson(0.1, size=(6, 600))
    cfg = Config(window_size=50, num_epochs=1, batch_size=8)
    res = infer_connectivity(spikes, native_bin_ms=1.0, bin_ms=1.0, n_runs=2,
                             ablation_threshold=-np.inf, cfg=cfg, verbose=False,
                             work_dir=tmp_path)

    assert res.A.shape == (6, 6)
    assert np.allclose(np.diagonal(res.A), 0.0)
    assert res.runs.shape == (2, 6, 6)
    assert res.ablation.shape == (2,)
    assert res.run_ids == [1, 2]
    assert res.bin_ms == 1.0
    assert res.downsample_factor == 1
    assert np.array_equal(res.A, np.round(res.runs.mean(axis=0), 3))
    assert np.array_equal(res.A_consistent, res.A * res.consistency)
    assert set(np.unique(res.consistency)) <= {0.0, 1.0}

    # the replicates written to work_dir read back as trained (up to each run's sign flip)
    matrices, ablation, run_ids = load_runs(tmp_path)
    assert run_ids == [1, 2]
    assert np.allclose(ablation, res.ablation)
    for m, aligned in zip(matrices, res.runs):
        assert np.allclose(np.abs(m), np.abs(aligned))
    assert all((tmp_path / f"best_model_run_{i}.pt").exists() for i in run_ids)

    res.save(tmp_path / "result.npz")
    with np.load(tmp_path / "result.npz") as saved:
        assert np.array_equal(saved["A"], res.A)
        assert saved["run_ids"].tolist() == [1, 2]


def test_infer_connectivity_rejects_bad_input():
    with pytest.raises(ValueError, match="at least 2 neurons"):
        infer_connectivity(np.zeros((1, 500)), 1.0, bin_ms=1.0)
    with pytest.raises(ValueError, match="2-D"):
        infer_connectivity(np.zeros((2, 2, 2)), 1.0, bin_ms=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        infer_connectivity(-np.ones((3, 500)), 1.0, bin_ms=1.0)
    with pytest.raises(ValueError, match="too short"):
        infer_connectivity(np.zeros((3, 50)), 1.0, bin_ms=1.0)

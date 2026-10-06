import csv

import numpy as np
import pytest

from circuit_gnn import consensus, get_consistent_connections, load_runs, orient_runs


def _cosine(a, b):
    return float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _true_A():
    A = np.zeros((5, 5))
    A[0, 1], A[1, 2], A[3, 4], A[4, 0] = 1.0, -2.0, 0.5, -0.7
    return A


def test_orient_runs_undoes_global_flips():
    A = _true_A()
    noise = np.random.default_rng(0).normal(0, 0.05, size=(3, 5, 5))
    runs = [A + noise[0], -(A + noise[1]), A + noise[2]]

    oriented, flip = orient_runs(runs, np.sign(A))
    assert flip == [False, True, False]
    assert all(_cosine(m, A) > 0.9 for m in oriented)

    # the data-derived sign decides the orientation of the set as a whole
    oriented, flip = orient_runs([-r for r in runs], np.sign(A))
    assert flip == [True, False, True]
    assert all(_cosine(m, A) > 0.9 for m in oriented)


def test_get_consistent_connections():
    stack = np.array([[[0, 1], [-1, 0]],
                      [[0, 1], [1, 0]],
                      [[0, 1], [-1, 0]]], dtype=float)
    assert get_consistent_connections(stack, 2 / 3).tolist() == [[0, 1], [1, 0]]
    assert get_consistent_connections(stack, 0.8).tolist() == [[0, 1], [0, 0]]


def test_consensus_gates_rounds_and_masks():
    spikes = np.random.default_rng(1).poisson(0.2, size=(5, 2000))
    A = _true_A() * 0.01234
    matrices = [A, 1.1 * A, 0.9 * A]

    c = consensus(matrices, [-0.5, 0.0, -0.3], spikes, run_ids=[7, 8, 9], ablation_threshold=0.1)
    assert c.run_ids == [7, 9]
    assert c.runs.shape == (2, 5, 5)
    assert np.array_equal(c.A, np.round(c.runs.mean(axis=0), 3))
    assert np.array_equal(c.A_consistent, c.A * c.consistency)
    assert set(c.flipped) <= {7, 9}

    raw = consensus(matrices, [-0.5, 0.0, -0.3], spikes, round_decimals=None)
    assert np.allclose(raw.A, raw.runs.mean(axis=0))


def test_consensus_keeps_every_run_when_all_fail():
    spikes = np.random.default_rng(2).poisson(0.2, size=(5, 2000))
    with pytest.warns(UserWarning, match="rejected all"):
        c = consensus([_true_A(), _true_A()], [0.0, 0.0], spikes)
    assert c.run_ids == [1, 2]


def test_load_runs_last_row_wins(tmp_path):
    for run_id in (1, 2):
        np.save(tmp_path / f"gnn_{run_id}.npy", np.full((2, 2), float(run_id)))
    with open(tmp_path / "ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "A_val_loss", "no_A_val_loss", "difference"])
        w.writerows([[2, 0.5, 0.6, -0.1], [1, 0.4, 0.7, -0.3], [2, 0.5, 0.9, -0.4]])

    matrices, ablation, run_ids = load_runs(tmp_path)
    assert run_ids == [1, 2]
    assert ablation.tolist() == [-0.3, -0.4]
    assert matrices[1][0, 0] == 2.0
    with pytest.raises(FileNotFoundError, match="missing"):
        load_runs(tmp_path, run_ids=[1, 3])

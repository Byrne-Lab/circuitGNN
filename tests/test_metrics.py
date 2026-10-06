import numpy as np
import pytest

from circuit_gnn import calculate_fit

TRUE = np.array([[0.0, 1.0, -2.0],
                 [0.0, 0.0, 0.5],
                 [0.0, 0.0, 0.0]])


def test_perfect_recovery_up_to_scale():
    res = calculate_fit(TRUE, 0.01 * TRUE)
    assert res.signed_precision == res.signed_recall == res.signed_f1 == 1.0
    assert res.cosine == pytest.approx(1.0)
    assert res.nie == pytest.approx(0.0, abs=1e-4)


def test_wrong_signs_are_misses():
    res = calculate_fit(TRUE, -TRUE)
    assert res.overall_f1 == 1.0       # every edge found ...
    assert res.signed_f1 == 0.0        # ... with the wrong sign

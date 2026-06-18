import numpy as np


def threshold_top_k(conn, true_conn):
    # keep the k strongest couplings, where k = number of true synapses
    off_diag = ~np.eye(conn.shape[0], dtype=bool)
    k = int(np.count_nonzero(true_conn[off_diag]))

    thresholded = np.zeros_like(conn)
    if k > 0:
        mag = np.abs(conn)
        mag[~off_diag] = -np.inf
        flat_idx = np.argsort(mag, axis=None)[::-1][:k]
        rows, cols = np.unravel_index(flat_idx, conn.shape)
        thresholded[rows, cols] = conn[rows, cols]
    return thresholded

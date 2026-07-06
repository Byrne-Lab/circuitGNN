"""Diagnose why sign_inference produces NaN for a dataset.

Checks the two failure modes of partial_corr / sign_inference:
  1. silent/constant neurons -> NaN rows in the correlation matrix
  2. ill-conditioned (near-singular) correlation matrix -> unstable inverse

Run from repo root, e.g.:
    python variants/diagnose_sign_inference.py --dataset mdp
"""

import os
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

VARIANTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = VARIANTS_DIR.parent
sys.path.insert(0, str(VARIANTS_DIR))
os.chdir(str(REPO_ROOT))

from compare_variants import DATASET_REGISTRY
from circuit_gnn.utils import sign_inference


def main():
    parser = argparse.ArgumentParser(description="Diagnose sign_inference NaN failures")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--num_taus", type=int, default=3)
    args = parser.parse_args()

    reg = DATASET_REGISTRY[args.dataset]
    binned_spikes, _, _ = reg["loader"](**reg["loader_kwargs"])
    N, T = binned_spikes.shape
    print(f"\n{args.dataset}: {N} neurons x {T} time bins")

    # 1. silent / constant neurons (theory: any std==0 -> NaN correlation row)
    std = binned_spikes.std(axis=1)
    silent = np.where(std == 0)[0]
    print(f"zero-variance neurons: {len(silent)} of {N}")
    if len(silent):
        print(f"  indices: {silent.tolist()[:20]}{' ...' if len(silent) > 20 else ''}")

    # 2. conditioning of the base correlation matrix
    C = np.corrcoef(binned_spikes)
    print(f"NaN in correlation matrix: {np.isnan(C).any()}")
    if not np.isnan(C).any():
        print(f"  condition number: {np.linalg.cond(C):.3e}")
        eig = np.linalg.eigvalsh(C)
        print(f"  min eigenvalue: {eig.min():.3e}  (negative/near-zero => not positive-definite)")

    # 3. confirm on the actual output sign_inference produces
    gdi_sign, _ = sign_inference(binned_spikes.T, args.num_taus)
    n_bad = int(np.sum(~np.isfinite(gdi_sign)))
    print(f"non-finite gdi_sign entries: {n_bad} of {gdi_sign.size} "
          f"({100 * n_bad / gdi_sign.size:.1f}%)")


if __name__ == "__main__":
    main()

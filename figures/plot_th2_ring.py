"""Sort the th-2 HD block by preferred direction and test for ring structure.

The learned matrix comes out in probe order, which scrambles any tuning-based
structure. This re-sorts the HD neurons by preferred head direction and asks
whether connection weight depends on the angular offset between cells.

Caveat this is built to expose: HD cells with similar preferred direction are
strongly co-active because they share one drive signal, so a peak at dPD = 0 is
predicted by BOTH a real ring and pure common-input contamination. The last
panel is the discriminator -- it regresses out tuning-curve correlation and
re-plots the profile on the residual.

    python figures/plot_th2_ring.py results/th2_A3716-201015b_wake1
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from circuit_gnn.utils import plot_region_matrix          # noqa: E402

SMOOTH_IDX = 3          # smoothTh index in HdTuning_*; 1 deg bins
N_DPD_BINS = 24         # 15 deg per bin
N_SHUFFLE = 200


def preferred_directions(tuning):
    """Circular mean and resultant length of each (n, 360) tuning curve."""
    theta = np.deg2rad(np.arange(360) + 0.5)
    v = tuning @ np.exp(1j * theta) / np.maximum(tuning.sum(1), 1e-12)
    return np.angle(v) % (2 * np.pi), np.abs(v)


def dpd_profile(A, pd, bins):
    """Mean weight binned by signed preferred-direction offset (post - pre)."""
    n = len(pd)
    d = (pd[None, :] - pd[:, None] + np.pi) % (2 * np.pi) - np.pi
    off = ~np.eye(n, dtype=bool)
    idx = np.digitize(d[off], bins) - 1
    w = A[off]
    out = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        m = idx == b
        if m.any():
            out[b] = w[m].mean()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--matrix", default="mean_consistent.npy")
    ap.add_argument("--session", default=None, help="default: parsed from results_dir")
    ap.add_argument("--epoch", default=None)
    ap.add_argument("--raw-root", default="datasets/th2/raw/Dataset_1")
    ap.add_argument("--prep-root", default="datasets/th2")
    args = ap.parse_args()

    base = os.path.basename(args.results_dir.rstrip("/\\"))
    m = re.match(r"th2_(.+?)_(wake\d|sleep\d)", base)
    session = args.session or (m and m.group(1))
    epoch = args.epoch or (m and m.group(2))
    if not session:
        sys.exit(f"could not parse session from {base!r}; pass --session/--epoch")

    A = np.load(os.path.join(args.results_dir, args.matrix))
    names = np.load(os.path.join(args.prep_root, f"prep_{session}_{epoch}", "pop_labels.npy"))
    groups = np.array([str(s).split("_")[0] for s in names])
    orig = np.array([int(str(s).split("_u")[-1]) - 1 for s in names])
    if A.shape[0] != len(names):
        sys.exit(f"matrix is {A.shape} but {len(names)} labels")

    td = sio.loadmat(os.path.join(args.raw_root, session, "Analysis",
                                  "HdTuning_xval_moveEp.mat"), squeeze_me=True)
    h = 0.5 * (np.asarray(td["hAll1"])[:, :, SMOOTH_IDX]
               + np.asarray(td["hAll2"])[:, :, SMOOTH_IDX])   # (360, n_all)
    tuning = h.T[orig]                                        # (n, 360)
    pd, mvl = preferred_directions(tuning)

    hd = np.flatnonzero(groups == "HD")
    print(f"{session} {epoch}: {len(names)} neurons, {len(hd)} HD")
    print(f"  HD resultant length: median {np.median(mvl[hd]):.3f} "
          f"(low values mean the preferred direction is poorly defined)")

    # per-block density, so the HD block is not judged by its area
    print("\n  block densities (nonzero / possible):")
    for a in ["HD", "FS", "EX"]:
        for b in ["HD", "FS", "EX"]:
            i, j = np.flatnonzero(groups == a), np.flatnonzero(groups == b)
            if not i.size or not j.size:
                continue
            sub = A[np.ix_(i, j)]
            poss = i.size * (i.size - 1) if a == b else i.size * j.size
            nz = int((sub != 0).sum())
            print(f"    {a}->{b}: {nz:5d}/{poss:5d} = {100 * nz / poss:5.2f}%  "
                  f"mean|w| {np.abs(sub[sub != 0]).mean() if nz else 0:.4f}")

    # --- 1. matrix re-sorted with HD ordered by preferred direction ---------
    order = np.concatenate([hd[np.argsort(pd[hd])],
                            np.flatnonzero(groups == "FS"),
                            np.flatnonzero(groups == "EX")])
    plot_region_matrix(A[np.ix_(order, order)], groups[order], ["HD", "FS", "EX"],
                       method_name="sorted_by_pd", save_dir=args.results_dir,
                       title=f"{session} {epoch} — HD sorted by preferred direction")

    # --- 2. weight vs preferred-direction offset, with shuffle null ---------
    A_hd = A[np.ix_(hd, hd)]
    pd_hd = pd[hd]
    bins = np.linspace(-np.pi, np.pi, N_DPD_BINS + 1)
    centers = np.rad2deg(0.5 * (bins[:-1] + bins[1:]))
    obs = dpd_profile(A_hd, pd_hd, bins)

    rng = np.random.default_rng(0)
    null = np.stack([dpd_profile(A_hd, rng.permutation(pd_hd), bins)
                     for _ in range(N_SHUFFLE)])
    lo, hi = np.nanpercentile(null, [2.5, 97.5], axis=0)

    # --- 3. the discriminator: same profile after removing tuning similarity -
    tc = np.corrcoef(tuning[hd])
    off = ~np.eye(len(hd), dtype=bool)
    x, y = tc[off], A_hd[off]
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    resid = np.zeros_like(A_hd)
    resid[off] = y - (slope * x + intercept)
    obs_resid = dpd_profile(resid, pd_hd, bins)
    print(f"\n  corr(weight, tuning-curve correlation) over HD pairs: r = {r:.3f}")
    print(f"  -> tuning similarity explains {100 * r ** 2:.1f}% of the variance "
          f"in HD->HD weights")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].fill_between(centers, lo, hi, color="0.85", label="shuffled PD, 95%")
    axes[0].plot(centers, obs, "k-o", ms=3, label="observed")
    axes[0].set_title("HD→HD weight vs ΔPD")
    axes[1].scatter(x, y, s=2, alpha=0.15, color="k")
    axes[1].plot(np.sort(x), slope * np.sort(x) + intercept, "r-", lw=1)
    axes[1].set_xlabel("tuning curve correlation")
    axes[1].set_ylabel("learned weight")
    axes[1].set_title(f"confound check: r = {r:.3f}")
    axes[2].axhline(0, color="0.7", lw=0.8)
    axes[2].plot(centers, obs_resid, "b-o", ms=3)
    axes[2].set_title("residual after removing tuning similarity")
    for ax in (axes[0], axes[2]):
        ax.set_xlabel("ΔPD (deg, post − pre)")
        ax.set_ylabel("mean weight")
        ax.set_xticks([-180, -90, 0, 90, 180])
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(args.results_dir, "ring_profile.pdf")
    plt.savefig(out, format="pdf")
    plt.close()
    print(f"\nWrote sorted_by_pd.pdf and ring_profile.pdf to {args.results_dir}/")


if __name__ == "__main__":
    main()

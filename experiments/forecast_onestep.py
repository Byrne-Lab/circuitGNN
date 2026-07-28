"""One-step forecast test for every circuitGNN checkpoint in a folder.

Every input window is real data, and only the next bin is predicted. Stacking the per-bin
predictions gives a spike train to compare against empirical FC.

Usage:
    python experiments/forecast_onestep.py /results/arbitrary
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

DPI = 600  # raster resolution of the rasterized=True imshow layers inside the vector PDF

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": DPI, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["arbitrary"]
plot_lim=0.04

def import_pkg():
    sys.path.insert(0, str(ROOT / "src"))
    os.chdir(ROOT)  # loaders resolve datasets/ relative to the repo root
    from circuit_gnn import data, models
    return data, models


def build_from_ckpt(state, models):
    # every hyperparameter is recoverable from the tensor shapes; buffers (rate_mean,
    # rate_denom) come back with load_state_dict, so eps/firing_rates need not be known
    enc, gnn_sd = state["encoder_state_dict"], state["gnn_state_dict"]
    num_neurons, mem_dim = enc["neuron_memory"].shape
    conv_out, _, kernel = enc["conv1d.weight"].shape
    emb_dim, fc_in = enc["fc.weight"].shape
    window = (fc_in - mem_dim) // conv_out
    out_dim, hidden = gnn_sd["fc_out.weight"].shape
    layers = 1 + max(int(k.split(".")[1]) for k in gnn_sd if k.startswith("layers."))

    encoder = models.SpikeEncoder(window_size=window, num_neurons=num_neurons,
                                  firing_rates=None, conv_out_channels=conv_out,
                                  embedding_dim=emb_dim, kernel_size=kernel, mem_dim=mem_dim)
    gnn = models.GraphNeuralNetwork(num_neurons=num_neurons, embedding_dim=emb_dim,
                                    hidden_dim=hidden, output_dim=out_dim, num_layers=layers)
    encoder.load_state_dict(enc)
    gnn.load_state_dict(gnn_sd)
    return encoder.eval(), gnn.eval(), window


def corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def offdiag(fc):
    return fc[np.triu_indices(fc.shape[0], k=1)]
 

def plot_diamond(fc_emp, fc_model, r, title, path, gap=0.06):
    n = fc_emp.shape[0]

    # each triangle keeps its own diagonal (self-correlation)
    # gap between two matrices comes from dx
    lower = np.tri(n, k=0, dtype=bool)  # create a triangle incl the diagonal
    emp = np.where(lower, fc_emp, np.nan) # keep emp's lower triangle
    mod = np.where(lower.T, fc_model, np.nan) # keep model's upper triangle
    dx = gap * n

    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    kw = dict(cmap="RdBu_r", vmin=-plot_lim, vmax=plot_lim, origin="lower", interpolation="nearest", rasterized=True)
    rot = mtransforms.Affine2D().rotate_deg(45) # 45 deg counter-clockwise
    ax.imshow(emp, transform=rot + mtransforms.Affine2D().translate(-dx, 0) + ax.transData, **kw)
    im = ax.imshow(mod, transform=rot + mtransforms.Affine2D().translate(dx, 0) + ax.transData, **kw)

    d, off = 2 ** -0.5, 0.04 * n
    ticks = sorted(set(list(range(0, n, max(1, round(n / 10)))) + [n - 1]))
    for k in ticks:
        ax.text(-dx - d * k - off, d * k - off, str(k),      # empirical (left) edge
                ha="right", va="center", rotation=-45, fontsize=6, color="0.35")
        ax.text(dx + d * k + off, d * k - off, str(k),       # model (right) edge
                ha="left", va="center", rotation=45, fontsize=6, color="0.35")

    # the diamond spans x = +-0.71n (+-dx each side); labels sit clear of its corners
    ax.set(xlim=(-(1.2 * n + dx), 1.2 * n + dx), ylim=(-0.2 * n, 1.52 * n), title=title)
    ax.axis("off")
    ax.text(-(0.98 * n + dx), 0.71 * n, "Empirical\nFC", ha="right", va="center", fontsize=9)
    ax.text(0.98 * n + dx, 0.71 * n, "Model\nFC", ha="left", va="center", fontsize=9)
    ax.text(0, -0.1 * n, f"r = {r:.2f}", ha="center", va="center", fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, label="correlation", ticks=[-plot_lim, 0, plot_lim])
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_heatmap(fc, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(fc, cmap="RdBu_r", vmin=-plot_lim, vmax=plot_lim, rasterized=True)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation", ticks=[-plot_lim, 0, plot_lim])
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def predict_next(encoder, gnn, spikes, start, window, device, batch=256):
    # every window is real data; only the next bin is predicted. unfold gives a view of
    # all windows, costs no extra memory until each batch is moved to the device.
    windows = spikes.unfold(1, window, 1).permute(1, 0, 2)  # (num_windows, N, W)
    windows = windows[start:spikes.shape[1] - window]       # predicts bins start+W .. T-1
    out = []
    for i in range(0, len(windows), batch):
        chunk = windows[i:i + batch].contiguous().to(device)
        out.append(torch.bernoulli(torch.sigmoid(gnn(encoder(chunk)))).cpu())
    return torch.cat(out).T.numpy()  # (N, n_steps)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, help="results folder, e.g. results/arbitrary/eps_1e-4")
    p.add_argument("--dataset", choices=DATASETS, help="default: inferred from the folder path")
    p.add_argument("--batch", type=int, default=256, help="windows per forward pass (default: 256)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    folder = args.folder.resolve()
    ckpts = sorted(folder.glob("*.pt"))
    if not ckpts:
        p.error(f"no .pt files in {folder}")
    data, models = import_pkg()

    name = args.dataset or next((d for d in DATASETS if d in folder.parts), None)
    if name is None:
        p.error("could not infer dataset from path; pass --dataset")
    binned, _, _ = getattr(data, f"load_{name}_network")(downsample_factor="auto")
    spikes = torch.tensor(binned, dtype=torch.float32)

    fc_emp, fcs, rs = None, [], []
    for i, ckpt in enumerate(ckpts):
        encoder, gnn, window = build_from_ckpt(torch.load(ckpt, map_location="cpu"), models)
        encoder, gnn = encoder.to(args.device), gnn.to(args.device)

        # only the held-out val split: same split as make_dataloaders
        start = int(0.8 * (spikes.shape[1] - window))
        if fc_emp is None:
            true = spikes[:, start + window:].numpy().clip(max=1.0)
            fc_emp = np.corrcoef(true)
            print(f"{name}: {spikes.shape[0]} neurons, window={window}, {len(ckpts)} checkpoints, "
                  f"predicting {true.shape[1]} held-out bins")

        torch.manual_seed(i)
        fc_gen = np.corrcoef(predict_next(encoder, gnn, spikes, start, window,
                                           args.device, args.batch))
        r = corr(offdiag(fc_emp), offdiag(fc_gen))
        fcs.append(fc_gen)
        rs.append(r)
        print(f"  {ckpt.name}: FC r = {r:.3f}")

    rs = np.array(rs)
    mean_r = np.nanmean(rs)
    print(f"\n  FC r (mean over {len(rs)}): {mean_r:.3f} +/- {np.nanstd(rs):.3f}")

    out = folder / "forecast_onestep"
    fc_mean = np.mean(fcs, axis=0)
    np.savez(out.with_suffix(".npz"), fc_r=rs, fc_emp=fc_emp, fc_gen=np.array(fcs), checkpoints=[c.name for c in ckpts])
    
    plot_diamond(fc_emp, fc_mean, mean_r, f"{name} / {folder.name} one-step ({len(rs)} runs, held-out)", out.with_suffix(".pdf"))
    
    plot_heatmap(fc_emp, f"{out}_emp.pdf")
    plot_heatmap(fc_mean, f"{out}_model.pdf")


if __name__ == "__main__":
    main()
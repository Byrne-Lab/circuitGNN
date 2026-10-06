"""Connectivity inference for a spike recording.

The model fits the adjacency matrix as a parameter of the recording it is trained on, so
there are no pretrained weights: ``fit_runs`` trains independent replicates on your data,
``consensus`` combines them, and ``infer_connectivity`` does both in one call.
"""
import csv
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .binning import rebin, resolve_downsample
from .consensus import (ABLATION_THRESHOLD, CONSISTENCY_THRESHOLD, ROUND_DECIMALS, SIGN_BIN_MS,
                        Consensus, consensus)
from .train import Config, ablation_losses, make_criterion, train_model


def fit_runs(spikes, *, n_runs=10, run_ids=None, cfg=None, out_dir=None, device=None,
             on_run=None, verbose=True):
    """Train independent replicates on one binned recording.

    spikes  : (N, T) non-negative spike counts at the analysis bin
    n_runs  : number of replicates; ignored when run_ids is given
    run_ids : ids of the replicates to train (default 1..n_runs)
    cfg     : Config with the model and training hyperparameters
    out_dir : if given, write gnn_<id>.npy, best_model_run_<id>.pt and a row of
              ablation.csv per replicate there (read them back with load_runs)
    on_run  : optional callback on_run(run_id, A), called after each replicate

    Returns ``(matrices, ablation, run_ids)``: each replicate's learned connectivity
    ([pre, post], zero diagonal), and its validation loss with A minus without A
    (negative when the learned graph helps prediction).
    """
    cfg = cfg or Config()
    device = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    run_ids = list(range(1, n_runs + 1)) if run_ids is None else list(run_ids)

    spike_tensor = torch.tensor(np.asarray(spikes), dtype=torch.float32)
    num_neurons, num_time = spike_tensor.shape
    if verbose:
        print(f"Loaded {num_neurons} channels, {num_time} time points.")
    criterion = make_criterion(spike_tensor, cfg.window_size, device)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "ablation.csv"
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["run_id", "A_val_loss", "no_A_val_loss", "difference"])

    matrices, ablation = [], []
    for run_id in run_ids:
        if verbose:
            print(f"\nRUN {run_id} (of {run_ids})")
        start = time.perf_counter()

        result = train_model(spike_tensor, cfg, criterion, device, verbose=verbose,
                             checkpoint_path=None if out_dir is None else out_dir / f"best_model_run_{run_id}.pt")
        encoder, gnn = result["encoder"], result["gnn"]   # weights of the best validation epoch
        with torch.no_grad():
            A = gnn.signed_adjacency().cpu().numpy().T     # [pre, post]
        if out_dir is not None:
            np.save(out_dir / f"gnn_{run_id}.npy", A)
        if on_run is not None:
            on_run(run_id, A)

        with_A, without_A = ablation_losses(encoder, gnn, result["val_loader"], criterion, device)
        if out_dir is not None:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([run_id, with_A, without_A, with_A - without_A])
        matrices.append(A)
        ablation.append(with_A - without_A)

        if verbose:
            print(f"[Run {run_id}] Validation loss (A):     {with_A:.6f}")
            print(f"[Run {run_id}] Validation loss (no A): {without_A:.6f}")
            print(f"Run {run_id} completed in {time.perf_counter() - start:.2f} seconds.")

    return matrices, np.asarray(ablation), run_ids


@dataclass
class InferenceResult(Consensus):
    ablation: np.ndarray      # (n_runs,) val loss with A minus without A, every replicate
    bin_ms: float             # width of the analysis bin actually used
    downsample_factor: int    # analysis bin in units of native_bin_ms
    neuron_names: np.ndarray

    def save(self, path):
        """Write every field to a compressed .npz file."""
        np.savez_compressed(path, **{f.name: np.asarray(getattr(self, f.name)) for f in fields(self)})


def infer_connectivity(
    spikes,
    native_bin_ms,
    *,
    bin_ms="auto",
    n_runs=10,
    ablation_threshold=ABLATION_THRESHOLD,
    consistency_threshold=CONSISTENCY_THRESHOLD,
    round_decimals=ROUND_DECIMALS,
    sign_bin_ms=SIGN_BIN_MS,
    cfg=None,
    neuron_names=None,
    device=None,
    work_dir=None,
    verbose=True,
) -> InferenceResult:
    """Infer signed connectivity from an (N, T) matrix of spike counts.

    spikes                : (N, T) non-negative counts at native_bin_ms resolution
    native_bin_ms         : width of one column of `spikes`, in milliseconds
    bin_ms                : analysis bin; 'auto' picks it by signed-edge stability
    n_runs                : independent replicates to train
    ablation_threshold    : keep a replicate only if using the graph lowers its validation
                            loss by at least this much (see consensus)
    consistency_threshold : fraction of kept replicates that must agree on an edge's sign
    round_decimals        : decimals the consensus mean is rounded to; None disables it
    sign_bin_ms           : bin used for the global sign estimate, independent of bin_ms
    cfg                   : Config with the model and training hyperparameters
    work_dir              : if given, every replicate is saved there (see fit_runs)
    """
    spikes = np.asarray(spikes)
    if spikes.ndim != 2:
        raise ValueError(f"spikes must be 2-D (neurons, time); got shape {spikes.shape}")
    if spikes.shape[0] < 2:
        raise ValueError(f"need at least 2 neurons; got {spikes.shape[0]}")
    if spikes.shape[1] == 0:
        raise ValueError("spikes has no time bins")
    if spikes.min() < 0:
        raise ValueError("spikes must be non-negative counts")

    cfg = cfg or Config()
    factor = resolve_downsample(
        bin_ms if isinstance(bin_ms, str) else max(1, round(bin_ms / native_bin_ms)),
        spikes, native_bin_ms, verbose=verbose)
    binned = rebin(spikes, factor) if factor > 1 else spikes

    num_time = binned.shape[1]
    if num_time - cfg.window_size < 10:
        raise ValueError(
            f"recording too short: {num_time} bins of {factor * native_bin_ms:g} ms leaves "
            f"{num_time - cfg.window_size} training windows at window_size={cfg.window_size}. "
            "Use a finer bin_ms, a shorter window_size, or a longer recording.")

    matrices, ablation, run_ids = fit_runs(binned, n_runs=n_runs, cfg=cfg, out_dir=work_dir,
                                           device=device, verbose=verbose)
    sign_spikes = rebin(spikes, max(1, round(sign_bin_ms / native_bin_ms)))
    c = consensus(matrices, ablation, sign_spikes, run_ids=run_ids,
                  ablation_threshold=ablation_threshold,
                  consistency_threshold=consistency_threshold, round_decimals=round_decimals)
    return InferenceResult(
        **vars(c),
        ablation=ablation,
        bin_ms=factor * native_bin_ms,
        downsample_factor=factor,
        neuron_names=np.arange(spikes.shape[0]) if neuron_names is None else np.asarray(neuron_names),
    )

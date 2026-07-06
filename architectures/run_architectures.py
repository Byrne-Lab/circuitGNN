"""Architecture experiment runner for circuitGNN.

Run from repo root:
    python architectures/run_architectures.py
    python architectures/run_architectures.py --datasets yuto --architectures baseline agg_vpa
    python architectures/run_architectures.py --datasets yuto pdcm --num_runs 10

Results are written to architectures/results/{dataset}/{architecture}/.
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

import torch
import numpy as np

# path setup: put architectures/ first so 'circuit_gnn' resolves to architectures/circuit_gnn/
ARCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = ARCH_DIR.parent
sys.path.insert(0, str(ARCH_DIR))
os.chdir(str(REPO_ROOT))

from circuit_gnn.data import (load_yuto_network, load_pdcm_network, load_pdcm_large_network,
                               load_arbitrary_network,
                               load_small_circuit_A_network, load_small_circuit_B_network,
                               load_mdp_network, load_v1_v2_network)
from circuit_gnn.train import TrainConfig, train_model, load_checkpoint
from circuit_gnn.utils import (plot_connectivity_matrix, plot_pdcm_matrix, plot_mdp_matrix,
                               plot_v1_v2_matrix, plot_edge_magnitude_vs_rate)

RESULTS_BASE = ARCH_DIR / "results"

_BASE_ENC = {"conv_out_channels": 32, "embedding_dim": 64, "kernel_size": 3}
_BASE_GNN = {"embedding_dim": 64, "hidden_dim": 64, "output_dim": 1,
             "num_layers": 5, "desired_max": 0.05, "init_noise": 0.01}

# compute_loss normalizes the adjacency penalties by N (row count), so these are
# the per-neuron-budget lambdas shared with src/ and variants/ (l1=7.5, comp=15).
_BASE_CFG = {"batch_size": 16, "lr": 1e-4, "l1_lambda": 7.5,
             "competitive_lambda": 15.0, "k_steps": 1}

ARCHITECTURES = {
    # 1 — baseline: signed scalar adjacency + 2-layer message MLP, weighted-sum aggregation
    "baseline": {
        "config": {**_BASE_CFG, "model_type": "standard"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # 2 — Hadamard modulator: additive + multiplicative (z_i ⊙ (A z)_i) paths (AMAG-style)
    "hadamard_mod": {
        "config": {**_BASE_CFG, "model_type": "hadamard_mod"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # 3 — multi-step: predict k=4 future bins simultaneously
    "multi_step": {
        "config": {**_BASE_CFG, "model_type": "multi_step", "k_steps": 4},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # 3b — message normalization: unit-norm each source's message before aggregation
    # so the edge weight A[i,j] alone carries influence, decoupled from source
    # activity (restores gradient to low-rate presynaptic edges). Parameter-free.
    "msg_norm": {
        "config": {**_BASE_CFG, "model_type": "msg_norm"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # 4 — aggregation axis (reduction of neighbour messages over the learned A)
    "agg_sum": {        # sum_j A[i,j] m_j  (== baseline aggregation)
        "config": {**_BASE_CFG, "model_type": "agg_sum"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    "agg_mean": {       # sum_j A[i,j] m_j / sum_j |A[i,j]|
        "config": {**_BASE_CFG, "model_type": "agg_mean"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    "agg_max": {        # elementwise max_j ( A[i,j] m_j )
        "config": {**_BASE_CFG, "model_type": "agg_max"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    "agg_vpa": {        # sum_j A[i,j] m_j / sqrt(sum_j A[i,j]^2)  (variance-preserving)
        "config": {**_BASE_CFG, "model_type": "agg_vpa"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    "agg_attention": {  # GAT attention replaces A; readout = mean attention matrix
        "config": {**_BASE_CFG, "model_type": "agg_attention"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # 5 — update axis (how the aggregated message integrates into the node state)
    "update_gru": {     # GRU cell mixes neighbour message into node state (GGS-NN)
        "config": {**_BASE_CFG, "model_type": "update_gru"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    "update_conv": {    # Conv1D over the feature dim mixes self + message
        "config": {**_BASE_CFG, "model_type": "update_conv"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # ENCODER axis — persistent learnable memory vector per neuron (baseline GNN)
    "memory_slot": {
        "config": {**_BASE_CFG, "model_type": "standard", "encoder_type": "memory_slot"},
        "encoder_kwargs": dict(**_BASE_ENC, mem_dim=16),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # ENCODER axis — per-neuron memory FiLM-modulates the conv features (gamma,beta)
    "memory_film": {
        "config": {**_BASE_CFG, "model_type": "standard", "encoder_type": "memory_film"},
        "encoder_kwargs": dict(**_BASE_ENC, mem_dim=16),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # ENCODER axis — memory_film with tanh-bounded scale: (1+gamma) in (0,2), stable
    "memory_film_bounded": {
        "config": {**_BASE_CFG, "model_type": "standard", "encoder_type": "memory_film_bounded"},
        "encoder_kwargs": dict(**_BASE_ENC, mem_dim=16),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # ENCODER axis — bounded FiLM modulation + concatenated memory (slot + film)
    "memory_hybrid": {
        "config": {**_BASE_CFG, "model_type": "standard", "encoder_type": "memory_hybrid"},
        "encoder_kwargs": dict(**_BASE_ENC, mem_dim=16),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # ENCODER axis — GRU over the window time axis, seeded by per-neuron memory
    # (dynamic recurrent state instead of a static prior)
    "memory_recurrent": {
        "config": {**_BASE_CFG, "model_type": "standard", "encoder_type": "memory_recurrent"},
        "encoder_kwargs": dict(**_BASE_ENC, mem_dim=16, gru_hidden=64, gru_layers=1),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # 6 — interpretability probe: trains exactly like baseline; adds Jacobian /
    # mono-vs-polysynaptic effective-connectivity probes (see models.py banner)
    "interpretable": {
        "config": {**_BASE_CFG, "model_type": "interpretable"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # 7 — regularization axis: L1 sparsity penalty weighted per presynaptic neuron
    # by inverse firing rate, so edges out of low-rate sources are shrunk less
    # (targets the low-firing-presynaptic miss, e.g. V1 in v1_v2). firing_rates are
    # computed from the data in train_model; gnn_kwargs may set rate_eps/weight_clip.
    "rate_aware": {
        "config": {**_BASE_CFG, "model_type": "rate_aware"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # ENCODER axis — rate normalization: standardize each neuron's window by its own
    # baseline rate before the conv, so a rare source's spike is as salient as a
    # frequent one and gradient flows onto its outgoing edges (the "inject signal"
    # complement to rate_aware). Baseline GNN + uniform penalty, so it isolates the
    # encoder effect. firing_rates are computed from the data in train_model.
    "rate_norm": {
        "config": {**_BASE_CFG, "model_type": "standard", "encoder_type": "rate_norm"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
}

# bin size: 'auto' uses the occupancy-based selector (circuit_gnn.optimal_bin_size)
DATASET_REGISTRY = {
    "yuto": {
        "loader": load_yuto_network,
        "loader_kwargs": {"downsample_factor": "auto", "gap": 0.20, "step": 1},
        "num_epochs": 20,
        "default_window": 100,
        "l1_lambda": 2.0,
        "competitive_lambda": 15.0,
    },
    "pdcm": {
        "loader": load_pdcm_network,
        "loader_kwargs": {"downsample_factor": "auto"},
        "num_epochs": 50,
        "default_window": 100,
        "l1_lambda": 2.5,
        "competitive_lambda": 10.0,
    },
    "pdcm_large": {
        "loader": load_pdcm_large_network,
        "loader_kwargs": {"downsample_factor": "auto"},
        "num_epochs": 50,
        "default_window": 100,
        "l1_lambda": 1.0,
        "competitive_lambda": 15.0,
    },
    "mdp": {
        "loader": load_mdp_network,
        "loader_kwargs": {"downsample_factor": "auto"},
        "num_epochs": 50,
        "default_window": 100,
        "l1_lambda": 1.0,
        "competitive_lambda": 15.0,
    },
    "v1_v2": {
        "loader": load_v1_v2_network,
        "loader_kwargs": {"downsample_factor": "auto"},
        "num_epochs": 50,
        "default_window": 100,
        "l1_lambda": 1.0,
        "competitive_lambda": 15.0,
    },
    "arbitrary": {
        "loader": load_arbitrary_network,
        "loader_kwargs": {"downsample_factor": "auto"},
        "num_epochs": 20,
        "default_window": 100,
        "l1_lambda": 7.5,
        "competitive_lambda": 5.0,
    },
    "small_circuit_A": {
        "loader": load_small_circuit_A_network,
        "loader_kwargs": {"downsample_factor": "auto"},
        "num_epochs": 20,
        "default_window": 100,
        "l1_lambda": 7.5,
        "competitive_lambda": 15.0,
    },
    "small_circuit_B": {
        "loader": load_small_circuit_B_network,
        "loader_kwargs": {"downsample_factor": "auto"},
        "num_epochs": 20,
        "default_window": 100,
        "l1_lambda": 1.5,
        "competitive_lambda": 15.0,
    },
}


def make_criterion(spike_tensor: torch.Tensor, device) -> torch.nn.Module:
    pos = spike_tensor.sum().item()
    neg = spike_tensor.numel() - pos
    pos_weight = torch.tensor(neg / pos, device=device) if pos > 0 else torch.tensor(1.0, device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def extract_adjacency(gnn, encoder=None, val_loader=None, device=None) -> np.ndarray:
    # attention variant: the connectivity readout is the sample-averaged attention,
    # so (re)populate it from the loaded weights over the val set before reading.
    if hasattr(gnn, "reset_attention") and val_loader is not None:
        encoder.eval(); gnn.eval()
        gnn.reset_attention()
        prev_scale = gnn.A_scale
        gnn.A_scale = 1.0
        with torch.no_grad():
            for inputs, _ in val_loader:
                gnn(encoder(inputs.to(device)))
        gnn.A_scale = prev_scale
    with torch.no_grad():
        return gnn.get_adjacency().cpu().numpy().T


def eval_with_scale(encoder, gnn, val_loader, criterion, device, A_scale: float) -> float:
    original = gnn.A_scale
    gnn.A_scale = A_scale
    total = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = gnn(encoder(inputs))
            total += criterion(outputs, targets).item() * inputs.size(0)
    gnn.A_scale = original
    return total / len(val_loader.dataset)


def run_dataset_architecture(
    dataset_name: str,
    arch_name: str,
    dataset_cfg: dict,
    arch_spec: dict,
    num_runs: int,
    device,
) -> None:
    output_dir = RESULTS_BASE / dataset_name / arch_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {dataset_name} data...")
    binned_spikes, true_conn, neuron_names = dataset_cfg["loader"](**dataset_cfg["loader_kwargs"])
    num_channels, num_time = binned_spikes.shape
    spike_tensor = torch.tensor(binned_spikes, dtype=torch.float32)
    acfg = arch_spec["config"]
    criterion = make_criterion(spike_tensor, device)

    window_size = acfg.get("window_size", dataset_cfg["default_window"])
    pop_labels = np.load(f'datasets/{dataset_name}/pop_labels.npy') if dataset_name in ('pdcm', 'pdcm_large') else None

    csv_path = output_dir / "ablation.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["run_id", "A_val_loss", "no_A_val_loss", "difference"])

    for run_idx in range(1, num_runs + 1):
        print(f"\n  Run {run_idx}/{num_runs}")
        t0 = time.perf_counter()

        config = TrainConfig(
            window_size=window_size,
            batch_size=acfg.get("batch_size", 16),
            num_epochs=dataset_cfg["num_epochs"],
            lr=acfg.get("lr", 1e-4),
            l1_lambda=dataset_cfg.get("l1_lambda", acfg.get("l1_lambda", 7.5)),
            competitive_lambda=dataset_cfg.get("competitive_lambda", acfg.get("competitive_lambda", 15.0)),
            k_steps=acfg.get("k_steps", 1),
            model_type=acfg.get("model_type", "standard"),
            encoder_type=acfg.get("encoder_type", "standard"),
            device=str(device),
            seed=run_idx,
            output_dir=str(output_dir),
        )

        result = train_model(
            spike_data=spike_tensor,
            num_neurons=num_channels,
            config=config,
            criterion=criterion,
            encoder_kwargs=arch_spec["encoder_kwargs"],
            gnn_kwargs=arch_spec["gnn_kwargs"],
            run_id=run_idx,
        )

        encoder, gnn = result["encoder"], result["gnn"]

        best_path = output_dir / f"best_model_run_{run_idx}.pt"
        if best_path.exists():
            load_checkpoint(str(best_path), encoder, gnn)

        val_loader = result["val_loader"]
        learned_A = extract_adjacency(gnn, encoder, val_loader, device)
        np.save(output_dir / f"gnn_{run_idx}.npy", learned_A)

        # diagnostic: outgoing edge magnitude vs firing rate (confirms/refutes the
        # low-rate-presynaptic failure mode). learned_A is [pre, post].
        firing_rates = spike_tensor.mean(dim=1).cpu().numpy()
        plot_edge_magnitude_vs_rate(learned_A, firing_rates, neuron_names=neuron_names,
                                    method_name=f"GNN {run_idx}", save_dir=str(output_dir))
        if dataset_name in ('pdcm', 'pdcm_large'):
            plot_pdcm_matrix(learned_A, pop_labels, method_name=f"GNN {run_idx}", save_dir=str(output_dir))
        elif dataset_name == 'mdp':
            plot_mdp_matrix(learned_A, neuron_names, method_name=f"GNN {run_idx}", save_dir=str(output_dir))
        elif dataset_name == 'v1_v2':
            plot_v1_v2_matrix(learned_A, neuron_names, method_name=f"GNN {run_idx}", save_dir=str(output_dir))
        else:
            plot_connectivity_matrix(learned_A, neuron_names=neuron_names,
                                     method_name=f"GNN {run_idx}", save_dir=str(output_dir))

        encoder.eval()
        gnn.eval()
        val_full = eval_with_scale(encoder, gnn, val_loader, criterion, device, A_scale=1.0)
        val_no_A = eval_with_scale(encoder, gnn, val_loader, criterion, device, A_scale=0.0)
        diff = val_full - val_no_A

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([run_idx, val_full, val_no_A, diff])

        elapsed = time.perf_counter() - t0
        print(f"  val(A)={val_full:.4f}  val(no-A)={val_no_A:.4f}  diff={diff:.4f}  [{elapsed:.1f}s]")


def main():
    parser = argparse.ArgumentParser(description="circuitGNN architecture experiments")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_REGISTRY.keys()),
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Datasets to run (default: all)")
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES.keys()),
                        choices=list(ARCHITECTURES.keys()),
                        help="Architectures to run (default: all)")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of random-seed runs per (dataset, architecture) pair")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override num_epochs for all datasets (useful for smoke tests)")
    args = parser.parse_args()

    if args.num_epochs is not None:
        for cfg in DATASET_REGISTRY.values():
            cfg["num_epochs"] = args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Architectures: {args.architectures}")
    print(f"Runs per config: {args.num_runs}")
    print(f"Results → {RESULTS_BASE}\n")

    for dataset_name in args.datasets:
        for arch_name in args.architectures:
            print(f"\n{'='*60}")
            print(f"  Dataset: {dataset_name}   Architecture: {arch_name}")
            print(f"{'='*60}")
            run_dataset_architecture(
                dataset_name=dataset_name,
                arch_name=arch_name,
                dataset_cfg=DATASET_REGISTRY[dataset_name],
                arch_spec=ARCHITECTURES[arch_name],
                num_runs=args.num_runs,
                device=device,
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()

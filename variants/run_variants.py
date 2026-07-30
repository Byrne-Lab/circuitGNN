"""Variant sweep runner for circuitGNN (rate-norm + memory-slot + Hadamard model).

Run from repo root:
    python variants/run_variants.py --datasets X --variants Y --num_runs Z

Results are written to variants/results/{dataset}/{variant}/.

The model is the src/circuit_gnn architecture (rate-normalized memory-slot encoder +
Hadamard-modulator GNN). This sweep varies only the two knobs that architecture adds
on top of the baseline: `eps` (the rate-norm Poisson floor) and `mem_dim` (the
per-neuron memory width). Each is swept one-at-a-time around the default
(eps=1e-3, mem_dim=16); the shared default point is eps_1e-3 == mem_16.
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

import torch
import numpy as np

VARIANTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = VARIANTS_DIR.parent
sys.path.insert(0, str(VARIANTS_DIR))
os.chdir(str(REPO_ROOT))

from circuit_gnn.data import load_OC_2_network, load_pdcm_network, load_pdcm_large_network, load_arbitrary_network, load_small_circuit_A_network, load_small_circuit_B_network
from circuit_gnn.train import TrainConfig, train_model, load_checkpoint
from circuit_gnn.utils import plot_connectivity_matrix, plot_pdcm_matrix

RESULTS_BASE = VARIANTS_DIR / "results"

# variant definitions
#   config          — TrainConfig field overrides (batch_size, log_sum_penalty_lambda, ...)
#   encoder_kwargs  — passed to SpikeEncoder (conv_out_channels, kernel_size, embedding_dim,
#                     mem_dim, eps); num_neurons + firing_rates are supplied by train_model
#   gnn_kwargs      — passed to GraphNeuralNetwork (embedding_dim, hidden_dim, output_dim, num_layers)

_BASE_ENC = {"conv_out_channels": 32, "embedding_dim": 64, "kernel_size": 3, "mem_dim": 16, "eps": 1e-3}
_BASE_GNN = {"embedding_dim": 64, "hidden_dim": 64, "output_dim": 1, "num_layers": 5}
_BASE_CFG = {"batch_size": 16, "log_sum_penalty_lambda": 2.0, "competitive_lambda": 15.0}

VARIANTS = {

    # eps sweep — rate-norm Poisson floor (mem_dim fixed at 16)
    "eps_1e-4": {"config": dict(**_BASE_CFG),
                 "encoder_kwargs": {**_BASE_ENC, "eps": 1e-4}, "gnn_kwargs": dict(**_BASE_GNN)},
    "eps_1e-3": {"config": dict(**_BASE_CFG),                       # == mem_16 (shared default)
                 "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "eps_1e-2": {"config": dict(**_BASE_CFG),
                 "encoder_kwargs": {**_BASE_ENC, "eps": 1e-2}, "gnn_kwargs": dict(**_BASE_GNN)},
    "eps_1e-1": {"config": dict(**_BASE_CFG),
                 "encoder_kwargs": {**_BASE_ENC, "eps": 1e-1}, "gnn_kwargs": dict(**_BASE_GNN)},

    # mem_dim sweep — per-neuron memory width (eps fixed at 1e-3)
    "mem_4":  {"config": dict(**_BASE_CFG),
               "encoder_kwargs": {**_BASE_ENC, "mem_dim": 4}, "gnn_kwargs": dict(**_BASE_GNN)},
    "mem_8":  {"config": dict(**_BASE_CFG),
               "encoder_kwargs": {**_BASE_ENC, "mem_dim": 8}, "gnn_kwargs": dict(**_BASE_GNN)},
    "mem_16": {"config": dict(**_BASE_CFG),                         # == eps_1e-3 (shared default)
               "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "mem_32": {"config": dict(**_BASE_CFG),
               "encoder_kwargs": {**_BASE_ENC, "mem_dim": 32}, "gnn_kwargs": dict(**_BASE_GNN)},
}

# dataset registry
# downsample_factor='auto' -> the bin is chosen by signkappa_interior inside each loader
# (circuit_gnn.optimal_bin_size).
DATASET_REGISTRY = {
    "OC_2": {
        "loader": load_OC_2_network,
        "loader_kwargs": {"downsample_factor": "auto"},
        "num_epochs": 20,
        "default_window": 100,
    },
    "arbitrary": {
        "loader": load_arbitrary_network,
        "loader_kwargs": {"downsample_factor": "auto"},
        "num_epochs": 20,
        "default_window": 100,
    },
    "pdcm": {
        "loader": load_pdcm_network,
        "loader_kwargs": {"downsample_factor": "auto"},
        "num_epochs": 50,
        "default_window": 100,
    },
    "pdcm_large": {
        "loader": load_pdcm_large_network,
        "loader_kwargs": {"downsample_factor": "auto"},
        "num_epochs": 50,
        "default_window": 100,
    },
    "small_circuit_A": {
        "loader": load_small_circuit_A_network,
        "loader_kwargs": {"downsample_factor": "auto"},
        "num_epochs": 20,
        "default_window": 100,
    },
    "small_circuit_B": {
        "loader": load_small_circuit_B_network,
        "loader_kwargs": {"downsample_factor": "auto"},
        "num_epochs": 20,
        "default_window": 100,
    },
}


# helpers

def make_criterion(spike_tensor: torch.Tensor, device) -> torch.nn.Module:
    pos = spike_tensor.sum().item()
    neg = spike_tensor.numel() - pos
    pos_weight = torch.tensor(neg / pos, device=device) if pos > 0 else torch.tensor(1.0, device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def extract_adjacency(gnn: torch.nn.Module) -> np.ndarray:
    # signed adjacency = relu(exc) - relu(inh), diagonal zeroed; returned as [pre, post]
    with torch.no_grad():
        A = torch.relu(gnn.learned_adj_exc) - torch.relu(gnn.learned_adj_inh)
        A = A * (1 - torch.eye(A.size(0), device=A.device))
        return A.cpu().numpy().T


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


# main run logic

def run_dataset_variant(
    dataset_name: str,
    variant_name: str,
    dataset_cfg: dict,
    variant_spec: dict,
    num_runs: int,
    device,
) -> None:
    output_dir = RESULTS_BASE / dataset_name / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {dataset_name} data...")
    binned_spikes, true_conn, neuron_names = dataset_cfg["loader"](**dataset_cfg["loader_kwargs"])
    num_channels, num_time = binned_spikes.shape
    spike_tensor = torch.tensor(binned_spikes, dtype=torch.float32)
    criterion = make_criterion(spike_tensor, device)

    window_size = variant_spec["config"].get("window_size", dataset_cfg["default_window"])
    pop_labels = np.load(f'datasets/{dataset_name}/pop_labels.npy') if dataset_name in ('pdcm', 'pdcm_large') else None

    # Initialise ablation CSV if needed
    csv_path = output_dir / "ablation.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["run_id", "A_val_loss", "no_A_val_loss", "difference"])

    for run_idx in range(1, num_runs + 1):
        print(f"\n  Run {run_idx}/{num_runs}")
        t0 = time.perf_counter()

        vcfg = variant_spec["config"]
        config = TrainConfig(
            window_size=window_size,
            batch_size=vcfg.get("batch_size", 16),
            num_epochs=vcfg.get("num_epochs", dataset_cfg["num_epochs"]),
            lr=vcfg.get("lr", 1e-4),
            log_sum_penalty_lambda=vcfg.get("log_sum_penalty_lambda", 2.0),
            competitive_lambda=vcfg.get("competitive_lambda", 15.0),
            device=str(device),
            seed=run_idx,
            output_dir=str(output_dir),
        )

        result = train_model(
            spike_data=spike_tensor,
            num_neurons=num_channels,
            config=config,
            criterion=criterion,
            encoder_kwargs=variant_spec["encoder_kwargs"],
            gnn_kwargs=variant_spec["gnn_kwargs"],
            run_id=run_idx,
        )

        encoder, gnn = result["encoder"], result["gnn"]

        # Load best checkpoint for this run
        best_path = output_dir / f"best_model_run_{run_idx}.pt"
        if best_path.exists():
            load_checkpoint(str(best_path), encoder, gnn)

        # Save learned adjacency and plot
        learned_A = extract_adjacency(gnn)
        np.save(output_dir / f"gnn_{run_idx}.npy", learned_A)
        if dataset_name in ('pdcm', 'pdcm_large'):
            plot_pdcm_matrix(learned_A, pop_labels, method_name=f"GNN {run_idx}", save_dir=str(output_dir))
        else:
            plot_connectivity_matrix(learned_A, neuron_names=neuron_names,
                                     method_name=f"GNN {run_idx}", save_dir=str(output_dir))

        # Ablation test
        encoder.eval()
        gnn.eval()
        val_loader = result["val_loader"]
        val_full = eval_with_scale(encoder, gnn, val_loader, criterion, device, A_scale=1.0)
        val_no_A = eval_with_scale(encoder, gnn, val_loader, criterion, device, A_scale=0.0)
        diff = val_full - val_no_A

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([run_idx, val_full, val_no_A, diff])

        elapsed = time.perf_counter() - t0
        print(f"  val(A)={val_full:.4f}  val(no-A)={val_no_A:.4f}  diff={diff:.4f}  [{elapsed:.1f}s]")


def main():
    parser = argparse.ArgumentParser(description="circuitGNN variant sweep")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_REGISTRY.keys()),
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Datasets to run (default: all)")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()),
                        help="Variants to run (default: all)")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of random-seed runs per (dataset, variant) pair")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Variants: {args.variants}")
    print(f"Runs per config: {args.num_runs}")
    print(f"Results → {RESULTS_BASE}\n")

    for dataset_name in args.datasets:
        for variant_name in args.variants:
            print(f"\n{'='*60}")
            print(f"  Dataset: {dataset_name}   Variant: {variant_name}")
            print(f"{'='*60}")
            run_dataset_variant(
                dataset_name=dataset_name,
                variant_name=variant_name,
                dataset_cfg=DATASET_REGISTRY[dataset_name],
                variant_spec=VARIANTS[variant_name],
                num_runs=args.num_runs,
                device=device,
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()

"""Architecture experiment runner for circuitGNN.

Run from repo root:
    python architectures/run_architectures.py
    python architectures/run_architectures.py --datasets yuto --architectures baseline dales
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

from circuit_gnn.data import (load_yuto_network, load_pdcm_network, load_arbitrary_network,
                               load_small_circuit_A_network, load_small_circuit_B_network)
from circuit_gnn.train import TrainConfig, train_model, load_checkpoint
from circuit_gnn.utils import plot_connectivity_matrix, plot_pdcm_matrix

RESULTS_BASE = ARCH_DIR / "results"

_BASE_ENC = {"conv_out_channels": 32, "embedding_dim": 64, "kernel_size": 3}
_BASE_GNN = {"embedding_dim": 64, "hidden_dim": 64, "output_dim": 1,
             "num_layers": 5, "desired_max": 0.05, "init_noise": 0.01}

_BASE_CFG = {"batch_size": 16, "lr": 1e-4, "l1_lambda": 5e-2,
             "competitive_lambda": 1e-1, "indirect_lambda": 0.0, "k_steps": 1}

ARCHITECTURES = {
    "baseline": {
        "config": {**_BASE_CFG, "model_type": "standard"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # presynaptic neuron can only be inhibitory or excitatory, not both
    "dales": {
        "config": {**_BASE_CFG, "model_type": "dales", "competitive_lambda": 0.0},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # remove indirect connections via an additional loss term
    "indirect_filter": {
        "config": {**_BASE_CFG, "model_type": "indirect_filter"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # skip connections between GNN layers
    "skip_residual": {
        "config": {**_BASE_CFG, "model_type": "skip"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # edge-conditioned messages (W_message receives A[i,j] alongside x_j)
    "edge_cond": {
        "config": {**_BASE_CFG, "model_type": "edge_cond"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # separate exc/inh aggregation paths
    "split_path": {
        "config": {**_BASE_CFG, "model_type": "split_path"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # predict 4 future time steps simultaneously
    "multi_step": {
        "config": {**_BASE_CFG, "model_type": "multi_step", "k_steps": 4},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # baseline with learnable b1/b2 scalars on neighbor and self terms
    "learned_scale": {
        "config": {**_BASE_CFG, "model_type": "learned_scale"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # AMAG-inspired: additive + Hadamard-modulator paths, shared adjacency, learnable b/b1/b2
    "hadamard_mod": {
        "config": {**_BASE_CFG, "model_type": "hadamard_mod"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # messages computed from pairwise embedding differences h_i - h_j
    "diff": {
        "config": {**_BASE_CFG, "model_type": "diff"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # Transformer-style row-wise softmax over incoming edges (learnable temperature)
    "softmax": {
        "config": {**_BASE_CFG, "model_type": "softmax"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # learnable per-neuron self-term scale vector
    "per_neuron_self": {
        "config": {**_BASE_CFG, "model_type": "per_neuron_self"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # GRU-cell state update integrating neighbor messages (Yoon 2025-inspired)
    "gated": {
        "config": {**_BASE_CFG, "model_type": "gated"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # AMAG-style sample-dependent Adaptor S[b,i,j] modulating the shared adjacency
    "adaptive": {
        "config": {**_BASE_CFG, "model_type": "adaptive"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
    # baseline with a ReLU on the message path too (symmetric with W_self)
    "relu_message": {
        "config": {**_BASE_CFG, "model_type": "relu_message"},
        "encoder_kwargs": dict(**_BASE_ENC),
        "gnn_kwargs": dict(**_BASE_GNN),
    },
}

DATASET_REGISTRY = {
    "yuto": {
        "loader": load_yuto_network,
        "loader_kwargs": {"downsample_factor": 20, "gap": 0.20, "step": 1},
        "num_epochs": 20,
        "default_window": 100,
    },
    "pdcm": {
        "loader": load_pdcm_network,
        "loader_kwargs": {"downsample_factor": 2},
        "num_epochs": 50,
        "default_window": 100,
    },
    "arbitrary": {
        "loader": load_arbitrary_network,
        "loader_kwargs": {"downsample_factor": 10},
        "num_epochs": 20,
        "default_window": 100,
    },
    "small_circuit_A": {
        "loader": load_small_circuit_A_network,
        "loader_kwargs": {"downsample_factor": 20},
        "num_epochs": 20,
        "default_window": 100,
    },
    "small_circuit_B": {
        "loader": load_small_circuit_B_network,
        "loader_kwargs": {"downsample_factor": 20},
        "num_epochs": 20,
        "default_window": 100,
    },
}


def make_criterion(spike_tensor: torch.Tensor, device) -> torch.nn.Module:
    pos = spike_tensor.sum().item()
    neg = spike_tensor.numel() - pos
    pos_weight = torch.tensor(neg / pos, device=device) if pos > 0 else torch.tensor(1.0, device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def extract_adjacency(gnn: torch.nn.Module) -> np.ndarray:
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
    criterion = make_criterion(spike_tensor, device)

    acfg = arch_spec["config"]
    window_size = acfg.get("window_size", dataset_cfg["default_window"])
    pop_labels = np.load('datasets/pdcm/pop_labels.npy') if dataset_name == 'pdcm' else None

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
            l1_lambda=acfg.get("l1_lambda", 5e-2),
            competitive_lambda=acfg.get("competitive_lambda", 1e-1),
            indirect_lambda=acfg.get("indirect_lambda", 0.0),
            k_steps=acfg.get("k_steps", 1),
            model_type=acfg.get("model_type", "standard"),
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

        learned_A = extract_adjacency(gnn)
        np.save(output_dir / f"gnn_{run_idx}.npy", learned_A)
        if dataset_name == 'pdcm':
            plot_pdcm_matrix(learned_A, pop_labels, method_name=f"GNN {run_idx}", save_dir=str(output_dir))
        else:
            plot_connectivity_matrix(learned_A, neuron_names=neuron_names,
                                     method_name=f"GNN {run_idx}", save_dir=str(output_dir))

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
    parser = argparse.ArgumentParser(description="circuitGNN architecture experiments")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_REGISTRY.keys()),
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Datasets to run (default: all)")
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES.keys()),
                        choices=list(ARCHITECTURES.keys()),
                        help="Architectures to run (default: all)")
    parser.add_argument("--num_runs", type=int, default=10,
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

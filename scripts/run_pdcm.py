import os
import sys
from pathlib import Path
import torch
import numpy as np
import time

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

import csv
from circuit_gnn.data import load_pdcm_network
from circuit_gnn.train import TrainConfig, train_model, load_checkpoint
from circuit_gnn.utils import plot_pdcm_matrix


def main():
    data = "pdcm"
    init_method = "none"
    signed_init = "signed" in init_method.lower()
    A_init_path = Path("datasets") / data / "init" / f"{init_method}.npy"

    downsample_factor = 1 # 1 ms
    num_runs = 50
    window_size = 100
    batch_size = 16
    num_epochs = 50
    lr = 1e-4
    l1_lambda = 5e-2
    competitive_lambda = 1e-1

    conv_filters = 32
    encoder_embedding_dim = 64
    gnn_hidden_dim = 64
    num_layers = 5
    output_dim = 1

    desired_max = 0.05
    init_noise = 0.01

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    binned_spikes, true_conn, _ = load_pdcm_network(downsample_factor=downsample_factor)
    pop_labels = np.load('datasets/pdcm/pop_labels.npy')
    num_channels, num_time = binned_spikes.shape
    print(f"Loaded {num_channels} channels, {num_time} time points.")

    A_init = None
    if A_init_path.exists():
        A_init = torch.tensor(np.load(A_init_path).T, dtype=torch.float32, device=device)
        print(f"Loaded adjacency init from {A_init_path}.")
    else:
        print(f"Warning: init file not found at {A_init_path}. Using random init.")

    if init_method.lower() == "none":
        init_type = "no_init"
    elif signed_init:
        init_type = "signed"
    else:
        init_type = "unsigned"

    output_dir = Path("results") / data / init_type
    output_dir.mkdir(parents=True, exist_ok=True)

    spike_tensor = torch.tensor(binned_spikes, dtype=torch.float32)

    # Compute pos_weight for BCE loss
    train_targets = spike_tensor[:, window_size:int(0.7 * num_time)]  # first 70% as train
    pos = train_targets.sum().item()
    neg = train_targets.numel() - pos
    pos_weight = torch.tensor(neg / pos, device=device) if pos > 0 else torch.tensor(1.0, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    all_A = []
    csv_path = output_dir / "ablation.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "A_val_loss", "no_A_val_loss", "difference"])

    for run_idx in range(1, num_runs + 1):
        print(f"\nRUN {run_idx} / {num_runs}")
        start_time = time.perf_counter()

        config = TrainConfig(
            window_size=window_size,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            l1_lambda=l1_lambda,
            competitive_lambda=competitive_lambda,
            device=device,
            seed=run_idx,  # if random split or shuffling in the future
            output_dir=str(output_dir),
        )

        result = train_model(
            spike_data=spike_tensor,
            num_neurons=num_channels,
            config=config,
            criterion=criterion,
            encoder_kwargs={"conv_out_channels": conv_filters, "embedding_dim": encoder_embedding_dim},
            gnn_kwargs={
                "embedding_dim": encoder_embedding_dim,
                "hidden_dim": gnn_hidden_dim,
                "output_dim": output_dim,
                "num_layers": num_layers,
                "A_init": A_init,
                "signed_init": signed_init,
                "desired_max": desired_max,
                "init_noise": init_noise,
            },
            run_id=run_idx,
        )

        print(f"Training complete for run {run_idx}.")

        # Extract learned adjacency
        encoder, gnn = result["encoder"], result["gnn"]

        # Load best model checkpoint for this run
        best_model_path = output_dir / f"best_model_run_{run_idx}.pt"
        if best_model_path.exists():
            load_checkpoint(str(best_model_path), encoder, gnn)
            print(f"Loaded best model from {best_model_path}.")
        with torch.no_grad():
            A_exc = torch.relu(gnn.learned_adj_exc)
            A_inh = torch.relu(gnn.learned_adj_inh)
            A_signed = A_exc - A_inh
            A_final = A_signed * (1 - torch.eye(A_signed.size(0), device=A_signed.device))
            learned_A = A_final.cpu().numpy().T

        np.save(output_dir / f"gnn_{run_idx}.npy", learned_A)
        plot_pdcm_matrix(learned_A, pop_labels, method_name=f"gnn_{run_idx}", save_dir=str(output_dir))
        all_A.append(learned_A)

        # Ablation test
        val_loader = result["val_loader"]
        encoder.eval()
        gnn.eval()

        # Baseline loss
        gnn.A_scale = 1.0
        val_full = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = encoder(inputs)
                outputs = gnn(logits)
                loss = criterion(outputs, targets)
                val_full += loss.item() * inputs.size(0)
        val_full /= len(val_loader.dataset)

        # Total ablation
        gnn.A_scale = 0.0
        val_no_A = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = encoder(inputs)
                outputs = gnn(logits)
                loss = criterion(outputs, targets)
                val_no_A += loss.item() * inputs.size(0)
        val_no_A /= len(val_loader.dataset)

        difference = val_full - val_no_A
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([run_idx, val_full, val_no_A, difference])

        print(f"[Run {run_idx}] Validation loss (A):     {val_full:.6f}")
        print(f"[Run {run_idx}] Validation loss (no A): {val_no_A:.6f}")

        end_time = time.perf_counter()
        print(f"Run {run_idx} completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch

from .train import TrainConfig, train_model, load_checkpoint


@dataclass
class RunConfig:
    # data / sweep
    downsample_factor: Union[int, str] = "auto"
    num_runs: int = 5
    init_method: str = "none"     

    # training
    window_size: int = 100
    batch_size: int = 16
    num_epochs: int = 20
    lr: float = 1e-4
    l1_lambda: float = 5e-2
    competitive_lambda: float = 1e-1

    # encoder / gnn
    conv_filters: int = 32
    embedding_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 5
    output_dim: int = 1
    desired_max: float = 0.05
    init_noise: float = 0.01


# plot_fn(learned_A, run_idx, save_dir, neuron_names) -> None
PlotFn = Callable[[np.ndarray, int, str, object], None]


def _extract_adjacency(gnn) -> np.ndarray:
    with torch.no_grad():
        A = torch.relu(gnn.learned_adj_exc) - torch.relu(gnn.learned_adj_inh)
        A = A * (1 - torch.eye(A.size(0), device=A.device))   # zero the diagonal
        return A.cpu().numpy().T                               # [pre, post]


def _eval_with_scale(encoder, gnn, val_loader, criterion, device, a_scale: float) -> float:
    gnn.A_scale = a_scale
    total = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = gnn(encoder(inputs))
            total += criterion(outputs, targets).item() * inputs.size(0)
    return total / len(val_loader.dataset)


def run_experiment(
    data: str,
    loader: Callable,
    cfg: Optional[RunConfig] = None,
    plot_fn: Optional[PlotFn] = None,
    loader_kwargs: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Train the baseline GNN for `cfg.num_runs` seeds and log the A ablation.

    data          : dataset name (used for the init/results paths)
    loader        : load_<dataset>_network; called with downsample_factor + loader_kwargs
    cfg           : RunConfig (universal defaults; pass overrides per script)
    plot_fn       : optional callback to render each run's learned adjacency
    loader_kwargs : extra kwargs for the loader (e.g. {'gap': 0.20, 'step': 1})
    """
    cfg = cfg or RunConfig()
    loader_kwargs = loader_kwargs or {}
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    signed_init = "signed" in cfg.init_method.lower()
    A_init_path = Path("datasets") / data / "init" / f"{cfg.init_method}.npy"

    binned_spikes, true_conn, neuron_names = loader(
        downsample_factor=cfg.downsample_factor, **loader_kwargs)
    num_channels, num_time = binned_spikes.shape
    print(f"Loaded {num_channels} channels, {num_time} time points.")

    A_init = None
    if A_init_path.exists():
        A_init = torch.tensor(np.load(A_init_path).T, dtype=torch.float32, device=device)
        print(f"Loaded adjacency init from {A_init_path}.")
    else:
        print(f"Warning: init file not found at {A_init_path}. Using random init.")

    if cfg.init_method.lower() == "none":
        init_type = "no_init"
    elif signed_init:
        init_type = "signed"
    else:
        init_type = "unsigned"

    output_dir = Path("results") / data / init_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # source is raw counts; the prediction target is clamped to {0,1}
    spike_tensor = torch.tensor(binned_spikes, dtype=torch.float32)

    # pos_weight for BCE, from the first 80% (train split)
    train_targets = spike_tensor[:, cfg.window_size:int(0.8 * num_time)]
    pos = train_targets.sum().item()
    neg = train_targets.numel() - pos
    pos_weight = torch.tensor(neg / pos, device=device) if pos > 0 else torch.tensor(1.0, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    csv_path = output_dir / "ablation.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["run_id", "A_val_loss", "no_A_val_loss", "difference"])

    for run_idx in range(1, cfg.num_runs + 1):
        print(f"\nRUN {run_idx} / {cfg.num_runs}")
        start_time = time.perf_counter()

        config = TrainConfig(
            window_size=cfg.window_size,
            batch_size=cfg.batch_size,
            num_epochs=cfg.num_epochs,
            lr=cfg.lr,
            l1_lambda=cfg.l1_lambda,
            competitive_lambda=cfg.competitive_lambda,
            device=device,
            seed=run_idx,
            output_dir=str(output_dir),
        )

        result = train_model(
            spike_data=spike_tensor,
            num_neurons=num_channels,
            config=config,
            criterion=criterion,
            encoder_kwargs={"conv_out_channels": cfg.conv_filters,
                            "embedding_dim": cfg.embedding_dim},
            gnn_kwargs={
                "embedding_dim": cfg.embedding_dim,
                "hidden_dim": cfg.hidden_dim,
                "output_dim": cfg.output_dim,
                "num_layers": cfg.num_layers,
                "A_init": A_init,
                "signed_init": signed_init,
                "desired_max": cfg.desired_max,
                "init_noise": cfg.init_noise,
            },
            run_id=run_idx,
        )
        print(f"Training complete for run {run_idx}.")

        encoder, gnn = result["encoder"], result["gnn"]
        best_model_path = output_dir / f"best_model_run_{run_idx}.pt"
        if best_model_path.exists():
            load_checkpoint(str(best_model_path), encoder, gnn)
            print(f"Loaded best model from {best_model_path}.")

        learned_A = _extract_adjacency(gnn)
        np.save(output_dir / f"gnn_{run_idx}.npy", learned_A)
        if plot_fn is not None:
            plot_fn(learned_A, run_idx, str(output_dir), neuron_names)

        encoder.eval()
        gnn.eval()
        val_loader = result["val_loader"]
        val_full = _eval_with_scale(encoder, gnn, val_loader, criterion, device, 1.0)
        val_no_A = _eval_with_scale(encoder, gnn, val_loader, criterion, device, 0.0)
        difference = val_full - val_no_A

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([run_idx, val_full, val_no_A, difference])

        print(f"[Run {run_idx}] Validation loss (A):     {val_full:.6f}")
        print(f"[Run {run_idx}] Validation loss (no A): {val_no_A:.6f}")
        print(f"Run {run_idx} completed in {time.perf_counter() - start_time:.2f} seconds.")

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import SpikeTrainDataset
from .models import GraphNeuralNetwork, SpikeEncoder


@dataclass
class TrainConfig:
    window_size: int = 100
    batch_size: int = 16
    num_epochs: int = 20
    lr: float = 1e-4
    l1_lambda: float = 5e-2    
    competitive_lambda: float = 1e-1   
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    num_workers: int = 0
    output_dir: str = "outputs"


def make_dataloaders(
    spike_data: torch.Tensor,
    window_size: int,
    batch_size: int,
    seed: int = 0,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if not isinstance(spike_data, torch.Tensor):
        spike_data = torch.tensor(spike_data, dtype=torch.float32)

    dataset = SpikeTrainDataset(spike_data, window_size)

    # 80% train, 20% val
    total_len = len(dataset)
    train_end = int(0.8 * total_len)
    train_set = torch.utils.data.Subset(dataset, range(0, train_end))
    val_set = torch.utils.data.Subset(dataset, range(train_end, total_len))
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_models(
    num_neurons: int,
    window_size: int,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    gnn_kwargs: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
) -> Tuple[SpikeEncoder, GraphNeuralNetwork]:
    encoder_kwargs = encoder_kwargs or {}
    gnn_kwargs = gnn_kwargs or {}

    encoder = SpikeEncoder(window_size=window_size, **encoder_kwargs).to(device)
    gnn = GraphNeuralNetwork(num_neurons=num_neurons, **gnn_kwargs).to(device)

    return encoder, gnn


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    l1_lambda: float = 5e-2,
    competitive_lambda: float = 1e-1,
    model: Optional[nn.Module] = None,
) -> torch.Tensor:
    loss = criterion(predictions, targets)
    if model is not None:
        # if l1_lambda > 0.0 and hasattr(model, 'learned_adj_exc') and hasattr(model, 'learned_adj_inh'):
        l1_adj = l1_lambda * (torch.log(1 + torch.abs(model.learned_adj_exc)).sum() + torch.log(1 + torch.abs(model.learned_adj_inh)).sum())
        # if competitive_lambda > 0.0 and hasattr(model, 'learned_adj_exc') and hasattr(model, 'learned_adj_inh'):
        competitive_penalty = competitive_lambda * (model.learned_adj_exc * model.learned_adj_inh).sum()
        loss = loss + l1_adj+ competitive_penalty
    return loss


def train_epoch(
    encoder: SpikeEncoder,
    gnn: GraphNeuralNetwork,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    competitive_lambda: float = 1e-1,
    l1_lambda: float = 5e-2,
) -> float:
    encoder.train()
    gnn.train()
    total_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        embeddings = encoder(inputs)
        outputs = gnn(embeddings)
        loss = compute_loss(outputs, targets, criterion, l1_lambda, competitive_lambda, model=gnn)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)


def eval_epoch(
    encoder: SpikeEncoder,
    gnn: GraphNeuralNetwork,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    encoder.eval()
    gnn.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            embeddings = encoder(inputs)
            outputs = gnn(embeddings)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)


def train_model(
    spike_data: torch.Tensor,
    num_neurons: int,
    config: TrainConfig,
    criterion: Optional[nn.Module] = None,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    gnn_kwargs: Optional[Dict[str, Any]] = None,
    run_id: Optional[int] = None,
) -> Dict[str, Any]:
    os.makedirs(config.output_dir, exist_ok=True)
    device = config.device

    train_loader, val_loader = make_dataloaders(
        spike_data,
        config.window_size,
        config.batch_size,
        config.seed,
        config.num_workers,
    )

    encoder, gnn = build_models(
        num_neurons=num_neurons,
        window_size=config.window_size,
        encoder_kwargs=encoder_kwargs,
        gnn_kwargs=gnn_kwargs,
        device=device,
    )

    if criterion is None:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(gnn.parameters()), lr=config.lr)

    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val = float('inf')

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_epoch(
            encoder,
            gnn,
            train_loader,
            optimizer,
            criterion,
            device,
            config.competitive_lambda,
            config.l1_lambda,
        )
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = eval_epoch(encoder, gnn, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            
            # Save best checkpoint
            if val_loss < best_val:
                best_val = val_loss
                model_name = f"best_model_run_{run_id}.pt" if run_id is not None else "best_model.pt"
                save_path = os.path.join(config.output_dir, model_name)
                save_checkpoint(save_path, encoder, gnn, optimizer, epoch, history)
        else:
            val_loss = None

        print(
            f"Epoch {epoch}/{config.num_epochs}: train_loss={train_loss:.6f}"
            + (f", val_loss={val_loss:.6f}" if val_loss is not None else "")
        )

    return {
        "encoder": encoder,
        "gnn": gnn,
        "history": history,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


def save_checkpoint(
    path: str,
    encoder: SpikeEncoder,
    gnn: GraphNeuralNetwork,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict[str, Any],
) -> None:
    state = {
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "gnn_state_dict": gnn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }
    torch.save(state, path)


def load_checkpoint(path: str, encoder: SpikeEncoder, gnn: GraphNeuralNetwork, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    encoder.load_state_dict(state["encoder_state_dict"])
    gnn.load_state_dict(state["gnn_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


if __name__ == "__main__":
    print("This module is intended to be imported. Use `train_model(...)` from your script.")

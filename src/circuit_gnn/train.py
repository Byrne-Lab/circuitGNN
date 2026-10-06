import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import GraphNeuralNetwork, SpikeEncoder


@dataclass
class Config:
    """Model and training hyperparameters; the defaults are the paper's."""
    # training
    window_size: int = 100              # input window, in bins
    batch_size: int = 16
    num_epochs: int = 20
    lr: float = 1e-4
    log_sum_penalty_lambda: float = 2.0  # sparsity weight
    competitive_lambda: float = 15.0     # excitatory/inhibitory competition weight

    # encoder
    conv_filters: int = 32
    embedding_dim: int = 64
    mem_dim: int = 32                   # width of each neuron's persistent memory slot
    eps: float = 1e-3                   # Poisson-scale floor for the encoder's rate normalization

    # gnn
    hidden_dim: int = 64
    num_layers: int = 5


# pytorch dataset for windows
class SpikeTrainDataset(torch.utils.data.Dataset):
    def __init__(self, spike_data, window_size): # called when dataset is created
        # spike_data: tensor of shape (num_channels, num_time)
        # window_size: number of time steps per window
        self.spike_data = spike_data
        self.window_size = window_size
        self.num_neurons, self.time_points = spike_data.shape
        self.num_windows = self.time_points - window_size

    def __len__(self): # returns the number of windows, len(dataset) = #
        return self.num_windows

    def __getitem__(self, index): # return index-th window, dataset[i] calls __getitem__, dataset[0] passes index = 0
        start = index
        end = start + self.window_size
        input_window = self.spike_data[:, start:end]
        target = self.spike_data[:, end].clamp(max=1.0)
        return input_window, target


def make_dataloaders(spike_data: torch.Tensor, window_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    dataset = SpikeTrainDataset(spike_data, window_size)

    # 80% train, 20% val
    total_len = len(dataset)
    train_end = int(0.8 * total_len)
    train_set = torch.utils.data.Subset(dataset, range(0, train_end))
    val_set = torch.utils.data.Subset(dataset, range(train_end, total_len))

    pin = torch.cuda.is_available()  # pinning only helps CUDA transfers
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=pin)
    return train_loader, val_loader


def make_criterion(spike_data: torch.Tensor, window_size: int, device) -> nn.Module:
    """Class-weighted BCE; pos_weight = negatives / positives in the training split.

    Counts are summed raw over bins [window_size, 0.8 T) while the dataset clamps its
    targets to {0, 1}; kept that way because the paper's replicates were trained with it.
    """
    num_time = spike_data.shape[1]
    train_targets = spike_data[:, window_size:int(0.8 * num_time)]
    pos = train_targets.sum().item()
    neg = train_targets.numel() - pos
    pos_weight = torch.tensor(neg / pos, device=device) if pos > 0 else torch.tensor(1.0, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def build_models(
    num_neurons: int,
    cfg: Config,
    device="cpu",
    firing_rates: Optional[torch.Tensor] = None,
) -> Tuple[SpikeEncoder, GraphNeuralNetwork]:
    # the encoder needs num_neurons (per-neuron memory + rate stats) and firing_rates
    # (per-neuron baseline for the rate normalization); firing_rates=None -> identity.
    encoder = SpikeEncoder(window_size=cfg.window_size, num_neurons=num_neurons,
                           firing_rates=firing_rates, conv_out_channels=cfg.conv_filters,
                           embedding_dim=cfg.embedding_dim, mem_dim=cfg.mem_dim,
                           eps=cfg.eps).to(device)
    gnn = GraphNeuralNetwork(num_neurons=num_neurons, embedding_dim=cfg.embedding_dim,
                             hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers).to(device)
    return encoder, gnn


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    log_sum_penalty_lambda: float = 2.0,
    competitive_lambda: float = 15.0,
    model: Optional[nn.Module] = None,
) -> torch.Tensor:
    loss = criterion(predictions, targets)
    if model is not None:

        penalty_norm = float(model.learned_adj_exc.shape[0]) # divide by N for a per-neuron budget

        log_sum_penalty_adj = log_sum_penalty_lambda / penalty_norm * (torch.log(1 + torch.abs(model.learned_adj_exc)).sum() + torch.log(1 + torch.abs(model.learned_adj_inh)).sum())
        competitive_penalty = competitive_lambda / penalty_norm * (torch.relu(model.learned_adj_exc) * torch.relu(model.learned_adj_inh)).sum()

        loss = loss + log_sum_penalty_adj + competitive_penalty

    return loss


def train_epoch(
    encoder: SpikeEncoder,
    gnn: GraphNeuralNetwork,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device,
    competitive_lambda: float = 15.0,
    log_sum_penalty_lambda: float = 2.0,
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
        loss = compute_loss(outputs, targets, criterion, log_sum_penalty_lambda, competitive_lambda, model=gnn)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)


def eval_epoch(
    encoder: SpikeEncoder,
    gnn: GraphNeuralNetwork,
    dataloader: DataLoader,
    criterion: nn.Module,
    device,
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


def ablation_losses(encoder, gnn, dataloader, criterion, device) -> Tuple[float, float]:
    """Validation loss of the model, and of the same model with its adjacency zeroed."""
    try:
        with_A = eval_epoch(encoder, gnn, dataloader, criterion, device)
        gnn.A_scale = 0.0
        without_A = eval_epoch(encoder, gnn, dataloader, criterion, device)
    finally:
        gnn.A_scale = 1.0
    return with_A, without_A


def train_model(
    spike_data,
    cfg: Config,
    criterion: nn.Module,
    device,
    checkpoint_path=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train one encoder + GNN on (N, T) spike counts.

    Returns dict(encoder, gnn, history, val_loader), with the weights of the epoch that had
    the lowest validation loss; that epoch is also saved to checkpoint_path, if given.
    """
    spike_data = torch.as_tensor(spike_data, dtype=torch.float32)
    train_loader, val_loader = make_dataloaders(spike_data, cfg.window_size, cfg.batch_size)

    # per-neuron mean spikes-per-bin for the encoder's rate normalization
    firing_rates = spike_data.float().mean(dim=1)
    encoder, gnn = build_models(spike_data.shape[0], cfg, device=device, firing_rates=firing_rates)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(gnn.parameters()), lr=cfg.lr)

    history: Dict[str, Any] = {"train_loss": [], "val_loss": []}
    best_val, best_state = float('inf'), None

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_epoch(encoder, gnn, train_loader, optimizer, criterion, device,
                                 cfg.competitive_lambda, cfg.log_sum_penalty_lambda)
        val_loss = eval_epoch(encoder, gnn, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy((encoder.state_dict(), gnn.state_dict()))
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, encoder, gnn, optimizer, epoch, history)

        if verbose:
            print(f"Epoch {epoch}/{cfg.num_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    if best_state is not None:
        encoder.load_state_dict(best_state[0])
        gnn.load_state_dict(best_state[1])

    return {"encoder": encoder, "gnn": gnn, "history": history, "val_loader": val_loader}


def save_checkpoint(
    path,
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


def load_checkpoint(path, encoder: SpikeEncoder, gnn: GraphNeuralNetwork, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["encoder_state_dict"])
    gnn.load_state_dict(state["gnn_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state

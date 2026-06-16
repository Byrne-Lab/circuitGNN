import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import SpikeTrainDataset, MultiStepSpikeTrainDataset
from .models import (GraphNeuralNetwork, SpikeEncoder,
                     DalesLawGNN, IndirectFilterGNN,
                     SkipGNN, EdgeCondGNN, SplitPathGNN, MultiStepGNN,
                     HadamardModGNN, LearnedScaleGNN, DiffGNN,
                     SoftmaxGNN, PerNeuronSelfScaleGNN, GatedGNN, AdaptiveGNN,
                     ReLUMessageGNN, ARResidualGNN)


@dataclass
class TrainConfig:
    window_size: int = 100
    batch_size: int = 16
    num_epochs: int = 20
    lr: float = 1e-4
    l1_lambda: float = 0.0
    competitive_lambda: float = 0.0
    reciprocity_lambda: float = 0.0
    indirect_lambda: float = 0.0
    k_steps: int = 1            # >1 activates MultiStepSpikeTrainDataset
    model_type: str = "standard"  # "standard"|"dales"|"indirect_filter"|"skip"|"edge_cond"|"split_path"|"multi_step"|"hadamard_mod"|"learned_scale"|"diff"|"softmax"|"per_neuron_self"|"gated"|"adaptive"|"relu_message"
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
    k_steps: int = 1,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if not isinstance(spike_data, torch.Tensor):
        spike_data = torch.tensor(spike_data, dtype=torch.float32)

    if k_steps > 1:
        dataset = MultiStepSpikeTrainDataset(spike_data, window_size, k_steps)
    else:
        dataset = SpikeTrainDataset(spike_data, window_size)

    total_len = len(dataset)
    train_end = int(0.8 * total_len)
    train_set = torch.utils.data.Subset(dataset, range(0, train_end))
    val_set = torch.utils.data.Subset(dataset, range(train_end, total_len))

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def build_models(
    num_neurons: int,
    window_size: int,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    gnn_kwargs: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    model_type: str = "standard",
    k_steps: int = 1,
) -> Tuple[SpikeEncoder, nn.Module]:
    encoder_kwargs = encoder_kwargs or {}
    gnn_kwargs = gnn_kwargs or {}

    encoder = SpikeEncoder(window_size=window_size, **encoder_kwargs).to(device)

    no_init_keys = ("A_init", "signed_init")

    if model_type == "dales":
        kw = {k: v for k, v in gnn_kwargs.items() if k not in no_init_keys}
        gnn = DalesLawGNN(num_neurons=num_neurons, **kw).to(device)
    elif model_type == "indirect_filter":
        gnn = IndirectFilterGNN(num_neurons=num_neurons, **gnn_kwargs).to(device)
    elif model_type == "skip":
        gnn = SkipGNN(num_neurons=num_neurons, **gnn_kwargs).to(device)
    elif model_type == "edge_cond":
        gnn = EdgeCondGNN(num_neurons=num_neurons, **gnn_kwargs).to(device)
    elif model_type == "split_path":
        gnn = SplitPathGNN(num_neurons=num_neurons, **gnn_kwargs).to(device)
    elif model_type == "multi_step":
        gnn = MultiStepGNN(num_neurons=num_neurons, k_steps=k_steps, **gnn_kwargs).to(device)
    elif model_type == "hadamard_mod":
        gnn = HadamardModGNN(num_neurons=num_neurons, **gnn_kwargs).to(device)
    elif model_type == "learned_scale":
        gnn = LearnedScaleGNN(num_neurons=num_neurons, **gnn_kwargs).to(device)
    elif model_type == "diff":
        gnn = DiffGNN(num_neurons=num_neurons, **gnn_kwargs).to(device)
    elif model_type == "softmax":
        gnn = SoftmaxGNN(num_neurons=num_neurons, **gnn_kwargs).to(device)
    elif model_type == "per_neuron_self":
        gnn = PerNeuronSelfScaleGNN(num_neurons=num_neurons, **gnn_kwargs).to(device)
    elif model_type == "gated":
        gnn = GatedGNN(num_neurons=num_neurons, **gnn_kwargs).to(device)
    elif model_type == "adaptive":
        gnn = AdaptiveGNN(num_neurons=num_neurons, **gnn_kwargs).to(device)
    elif model_type == "relu_message":
        gnn = ReLUMessageGNN(num_neurons=num_neurons, **gnn_kwargs).to(device)
    elif model_type == "ar_residual":
        gnn = ARResidualGNN(num_neurons=num_neurons, **gnn_kwargs).to(device)
    else:
        gnn = GraphNeuralNetwork(num_neurons=num_neurons, **gnn_kwargs).to(device)

    return encoder, gnn


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    l1_lambda: float = 5e-2,
    competitive_lambda: float = 1e-1,
    reciprocity_lambda: float = 0.0,
    indirect_lambda: float = 0.0,
    model: Optional[nn.Module] = None,
) -> torch.Tensor:
    # criterion handles any shape: (batch,N) for single-step, (batch,N,k) for multi-step
    loss = criterion(predictions, targets)
    if model is not None:
        if hasattr(model, 'learned_adj_exc') and hasattr(model, 'learned_adj_inh'):
            l1_adj = l1_lambda * (
                torch.log(1 + torch.abs(model.learned_adj_exc)).sum()
                + torch.log(1 + torch.abs(model.learned_adj_inh)).sum()
            )
            # penalize only coexisting rectified weights; raw params can go negative
            # (dead under the relu in get_adjacency), and a negative product would
            # *reward* growing the opposite matrix instead of penalizing overlap
            competitive_penalty = competitive_lambda * (
                torch.relu(model.learned_adj_exc) * torch.relu(model.learned_adj_inh)
            ).sum()
            loss = loss + l1_adj + competitive_penalty

            # reciprocity penalty: discourage a mirror edge pair where (i,j) is
            # excitatory and (j,i) is inhibitory (the antisymmetric "lead-lag"
            # artifact). Transpose of the competitive penalty's exc/inh coupling.
            if reciprocity_lambda > 0.0:
                reciprocity_penalty = reciprocity_lambda * (
                    torch.relu(model.learned_adj_exc) * torch.relu(model.learned_adj_inh).t()
                ).sum()
                loss = loss + reciprocity_penalty

        if indirect_lambda > 0.0 and hasattr(model, 'get_adjacency'):
            A = model.get_adjacency()
            A_two_hop = (A @ A).detach()
            loss = loss + indirect_lambda * (A * A_two_hop.abs()).abs().sum()

        # ar_residual: train the per-neuron self head to predict the target from
        # the neuron's own history alone, so the graph path only fits the residual.
        self_pred = getattr(model, 'last_self_pred', None)
        self_aux_lambda = getattr(model, 'self_aux_lambda', 0.0)
        if self_pred is not None and self_aux_lambda > 0.0:
            loss = loss + self_aux_lambda * criterion(self_pred, targets)

    return loss


def train_epoch(
    encoder: SpikeEncoder,
    gnn: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    competitive_lambda: float = 1e-1,
    l1_lambda: float = 5e-2,
    reciprocity_lambda: float = 0.0,
    indirect_lambda: float = 0.0,
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
        loss = compute_loss(
            outputs, targets, criterion,
            l1_lambda, competitive_lambda, reciprocity_lambda, indirect_lambda,
            model=gnn,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)


def eval_epoch(
    encoder: SpikeEncoder,
    gnn: nn.Module,
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

    # make each run reproducible: seeds model init and DataLoader shuffling
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_loader, val_loader = make_dataloaders(
        spike_data,
        config.window_size,
        config.batch_size,
        config.seed,
        config.num_workers,
        config.k_steps,
    )

    encoder, gnn = build_models(
        num_neurons=num_neurons,
        window_size=config.window_size,
        encoder_kwargs=encoder_kwargs,
        gnn_kwargs=gnn_kwargs,
        device=device,
        model_type=config.model_type,
        k_steps=config.k_steps,
    )

    if criterion is None:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(gnn.parameters()), lr=config.lr)

    history: Dict[str, Any] = {"train_loss": [], "val_loss": []}
    best_val = float('inf')

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_epoch(
            encoder, gnn, train_loader, optimizer, criterion, device,
            config.competitive_lambda,
            config.l1_lambda,
            config.reciprocity_lambda,
            config.indirect_lambda,
        )
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = eval_epoch(encoder, gnn, val_loader, criterion, device)
            history["val_loss"].append(val_loss)

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
    encoder: nn.Module,
    gnn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict[str, Any],
) -> None:
    torch.save({
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "gnn_state_dict": gnn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, path)


def load_checkpoint(
    path: str,
    encoder: nn.Module,
    gnn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    encoder.load_state_dict(state["encoder_state_dict"])
    gnn.load_state_dict(state["gnn_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state

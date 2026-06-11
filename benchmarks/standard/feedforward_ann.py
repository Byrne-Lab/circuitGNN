# Implementation of feed-forward ANN functional connectivity inference
# Based on: Frolov et al. (2019), Chaos 29, 091101, Pitsik et al. (2022), Sensors 22, 2537

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d


def _smooth_spikes(spikes: np.ndarray, sigma_bins: float) -> np.ndarray:
    return gaussian_filter1d(spikes.astype(np.float32), sigma=sigma_bins)


def _time_delay_embed(signal: np.ndarray, dim: int, tau: int) -> np.ndarray:
    n_vectors = len(signal) - (dim - 1) * tau
    return np.stack(
        [signal[i * tau: i * tau + n_vectors] for i in range(dim)],
        axis=1
    )


class FrolovMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.Softmax(dim=-1),          # hidden layer 1 — paper architecture
            nn.Linear(10, 10),
            nn.Softmax(dim=-1),          # hidden layer 2 — paper architecture
            nn.Linear(10, output_dim),   # linear output — paper architecture
        )

    def forward(self, x):
        return self.net(x)


def _train_mlp(
    X_states: np.ndarray,       # (N, D)
    Y_states: np.ndarray,       # (N, D)
    max_restarts: int = 3,
    epochs: int = 1000,
    divergence_patience: int = 10,
    device: torch.device = torch.device("cpu"),
) -> float:
    D_in  = X_states.shape[1]
    D_out = Y_states.shape[1]

    X = X_states.astype(np.float32)
    Y = Y_states.astype(np.float32)

    perm = np.random.permutation(len(X))
    X, Y = X[perm], Y[perm]

    split        = len(X) // 2
    X_tr, X_val  = X[:split], X[split:]
    Y_tr, Y_val  = Y[:split], Y[split:]

    X_tr_t  = torch.from_numpy(X_tr).to(device)
    Y_tr_t  = torch.from_numpy(Y_tr).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    Y_val_t = torch.from_numpy(Y_val).to(device)

    best_r2   = -np.inf
    criterion = nn.MSELoss()

    for attempt in range(max_restarts):
        model     = FrolovMLP(D_in, D_out).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        prev_gap         = None
        divergence_count = 0
        should_restart   = False

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            train_loss = criterion(model(X_tr_t), Y_tr_t)
            train_loss.backward()
            optimizer.step()
            train_loss_val = train_loss.item()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), Y_val_t).item()

            # Divergence check: restart if val-train gap grows for N epochs
            gap = val_loss - train_loss_val
            if prev_gap is not None and gap > prev_gap:
                divergence_count += 1
            else:
                divergence_count = 0
            prev_gap = gap
            if divergence_count >= divergence_patience:
                should_restart = True
                break

        model.eval()
        with torch.no_grad():
            Y_pred = model(X_val_t).cpu().numpy()

        ss_res  = np.sum((Y_val - Y_pred) ** 2)
        ss_tot  = np.sum((Y_val - Y_val.mean(axis=0)) ** 2)
        r2      = 1.0 - ss_res / (ss_tot + 1e-12)

        if r2 > best_r2:
            best_r2 = r2

        if not should_restart:
            break

    return float(best_r2)


def calculate_frolov_connectivity(
    binned_spikes: np.ndarray,
    embed_dim: int = 5,
    tau: int = 10,
    max_lag: int = 30,
    threshold: float = 0.5,
    max_restarts: int = 3,
    epochs: int = 1000,
    smooth_sigma: float = 0.0,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )

    if verbose:
        print(f"Frolov ANN  |  device={device}  embed_dim={embed_dim}  "
              f"tau={tau}  max_lag={max_lag}  smooth_sigma={smooth_sigma}")

    num_channels, num_time = binned_spikes.shape

    signals = (
        np.array([_smooth_spikes(binned_spikes[ch], smooth_sigma) for ch in range(num_channels)])
        if smooth_sigma > 0
        else binned_spikes.astype(np.float32)
    )

    if embed_dim > 1:
        states = [_time_delay_embed(signals[ch], embed_dim, tau) for ch in range(num_channels)]
        min_len = min(s.shape[0] for s in states)
        states  = [s[:min_len] for s in states]
    else:
        states = [signals[ch].reshape(-1, 1) for ch in range(num_channels)]

    r2_matrix   = np.zeros((num_channels, num_channels))
    total_pairs = num_channels * (num_channels - 1)
    done        = 0

    for src in range(num_channels):
        for tgt in range(num_channels):
            if src == tgt:
                continue

            best_r2_pair = -np.inf
            for lag in range(0, max_lag + 1):
                X_lag = states[src][:len(states[src]) - lag] if lag > 0 else states[src]
                Y_lag = states[tgt][lag:] if lag > 0 else states[tgt]

                r2 = _train_mlp(
                    X_states     = X_lag,
                    Y_states     = Y_lag,
                    max_restarts = max_restarts,
                    epochs       = epochs,
                    device       = device,
                )
                if r2 > best_r2_pair:
                    best_r2_pair = r2

            r2_matrix[src, tgt] = best_r2_pair
            done += 1

            if verbose:
                print(f"  [{done:>{len(str(total_pairs))}}/{total_pairs}]  "
                      f"ch{src} → ch{tgt}  best R²={best_r2_pair:.4f}")

    conn_matrix = np.where(r2_matrix > threshold, r2_matrix, 0.0)
    np.fill_diagonal(conn_matrix, 0.0)

    return r2_matrix, conn_matrix
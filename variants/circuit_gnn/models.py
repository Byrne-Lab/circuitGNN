import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any


class SpikeEncoder(nn.Module):
    def __init__(self, window_size, conv_out_channels=32, embedding_dim=64, kernel_size=3):
        super(SpikeEncoder, self).__init__()

        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(in_channels=1,
                                out_channels=conv_out_channels,
                                kernel_size=kernel_size,
                                padding=0)

        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(conv_out_channels)
        self.fc = nn.Linear(conv_out_channels * window_size, embedding_dim)

    def forward(self, x):
        batch_size, num_neurons, window_size = x.shape
        x = x.view(batch_size * num_neurons, 1, window_size)
        x = F.pad(x, (self.kernel_size - 1, 0))
        out = self.conv1d(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = out.view(batch_size * num_neurons, -1)
        embeddings = self.fc(out)
        embeddings = embeddings.view(batch_size, num_neurons, -1)
        return embeddings


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, self_scale=0.5):
        super(GNNLayer, self).__init__()

        mlp_dim = 2 * in_dim
        self.self_scale = self_scale

        self.W_message = nn.Sequential(
            nn.Linear(in_dim, mlp_dim, bias=True),
            nn.ReLU(),
            nn.Linear(mlp_dim, out_dim, bias=True),
        )
        self.W_self = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        messages = self.W_message(x)
        self_term = self.W_self(x)
        neighbor_agg = torch.matmul(A, messages)
        out = self.relu(self.norm(neighbor_agg + self.self_scale * self_term))
        return out


class GraphNeuralNetwork(nn.Module):
    def __init__(
        self,
        num_neurons,
        embedding_dim=64,
        hidden_dim=64,
        output_dim=1,
        num_layers=5,
        A_init=None,
        signed_init=False,
        desired_max=0.05,
        init_noise=0.1,
        self_scale=0.5,
    ):
        super(GraphNeuralNetwork, self).__init__()

        if A_init is not None and not signed_init:
            print("Using unsigned adjacency initialization.")
            A_max = A_init.abs().max()
            A_norm = (A_init / A_max) * desired_max
            A_noisy_init = A_norm + torch.randn_like(A_norm) * init_noise
            self.learned_adj_exc = nn.Parameter(A_noisy_init.clone())
            self.learned_adj_inh = nn.Parameter(A_noisy_init.clone())

        elif A_init is not None and signed_init:
            print('Using signed adjacency initialization.')
            N = A_init.shape[0]
            pos_mask = A_init > 0
            neg_mask = A_init < 0
            A_max = A_init.abs().max()
            A_norm_pos = (A_init[pos_mask] / A_max) * desired_max
            A_norm_neg = (torch.abs(A_init[neg_mask]) / A_max) * desired_max
            A_noisy_pos = A_norm_pos + torch.randn_like(A_norm_pos) * init_noise
            A_noisy_neg = A_norm_neg + torch.randn_like(A_norm_neg) * init_noise
            init_exc = torch.zeros(N, N, device=A_init.device)
            init_inh = torch.zeros(N, N, device=A_init.device)
            init_exc[pos_mask] = A_noisy_pos
            init_inh[neg_mask] = A_noisy_neg
            self.learned_adj_exc = nn.Parameter(init_exc.clone())
            self.learned_adj_inh = nn.Parameter(init_inh.clone())

        else:
            print("Using random numbers for adjacency initialization.")
            self.learned_adj_exc = nn.Parameter(0.01 * torch.randn(num_neurons, num_neurons).abs())
            self.learned_adj_inh = nn.Parameter(0.01 * torch.randn(num_neurons, num_neurons).abs())

        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(GNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        self.layers = nn.ModuleList(layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.A_scale = 1.0

    def get_adjacency(self) -> torch.Tensor:
        A_exc = torch.relu(self.learned_adj_exc)
        A_inh = torch.relu(self.learned_adj_inh)
        A = A_exc - A_inh
        return A * (1 - torch.eye(A.size(0), device=A.device))

    def forward(self, embeddings):
        A = self.get_adjacency() * self.A_scale
        x = embeddings
        for layer in self.layers:
            x = layer(x, A)
        return self.fc_out(x).squeeze(-1)
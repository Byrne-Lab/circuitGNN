import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any


class SpikeEncoder(nn.Module):

    def __init__(self, window_size, num_neurons, firing_rates=None,
                 conv_out_channels=32, embedding_dim=64, kernel_size=3,
                 mem_dim=16, eps=1e-3):
        # window_size: number of time steps in window
        # num_neurons: needed for the per-neuron rate stats and memory table
        # firing_rates: (num_neurons,) mean spikes/bin
        # conv_out_channels: number of Conv1D output channels per neuron
        # embedding_dim: size of final embedding vector per neuron
        # mem_dim: width of each neuron's persistent memory row
        # eps: Poisson-scale floor for the rate normalization
        super(SpikeEncoder, self).__init__()

        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(in_channels=1,
                                out_channels=conv_out_channels,
                                kernel_size=kernel_size,
                                padding=0 # no padding here
        )

        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(conv_out_channels) # normalize across feature maps
        self.neuron_memory = nn.Parameter(0.01 * torch.randn(num_neurons, mem_dim)) # persistent memory row per neuron, concatenated to neuron's flattened conv features
        self.fc = nn.Linear(conv_out_channels*window_size + mem_dim, embedding_dim) # fully connected layer that reduces the flattened convolution output (plus memory) to a fixed embedding vector

        # per-neuron standardization constants
        if firing_rates is None:
            mu = torch.zeros(num_neurons)
            denom = torch.ones(num_neurons)
        else:
            mu = torch.as_tensor(firing_rates, dtype=torch.float32).flatten()
            assert mu.numel() == num_neurons, \
                f"firing_rates has {mu.numel()} entries, expected {num_neurons}"
            denom = torch.sqrt(mu + eps) # Poisson scale; eps floors the amplification
        self.register_buffer("rate_mean", mu)
        self.register_buffer("rate_denom", denom)

    def forward(self, x):
        # x: (batch_size, num_neurons, window_size)
        batch_size, num_neurons, window_size = x.shape

        # standardize each neuron by its own baseline rate (broadcast over batch/time)
        x = (x - self.rate_mean.view(1, -1, 1)) / self.rate_denom.view(1, -1, 1)

        x = x.reshape(batch_size * num_neurons, 1, window_size) # (batch_size * num_neurons, 1, window_size)
        x = F.pad(x, (self.kernel_size - 1, 0))  # left-only causal pad
        out = self.conv1d(x) # (B*N, conv_out, window_size) : (B*N, 32, 100)

        out = self.batchnorm(out) # normalizes each of the 32 channels independently, computing mean and var across B*N batch and 100 time dim
        out = self.relu(out)

        out = out.view(batch_size, num_neurons, -1) # (B, N, conv_out * T) : (B, N, 3200)
        # append each neuron's persistent memory row, broadcast over the batch
        mem = self.neuron_memory.unsqueeze(0).expand(batch_size, -1, -1) # (B, N, mem_dim)
        out = torch.cat([out, mem], dim=-1) # (B, N, conv_out * T + mem_dim)

        embeddings = self.fc(out) # (B, N, embedding_dim) : (B, N, 64)

        return embeddings
    
class GNNLayer(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GNNLayer, self).__init__()

        mlp_dim = 2 * in_dim

        # MLPs for messages, Hadamard modulation, and self-connections
        self.W_message = nn.Sequential(
            nn.Linear(in_dim, mlp_dim, bias=True),
            nn.ReLU(),
            nn.Linear(mlp_dim, out_dim, bias=True),
        )
        self.W_mod = nn.Sequential(
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

        # learnable scalars for aggregation weighting
        self.b = nn.Parameter(torch.ones(1))
        self.b1 = nn.Parameter(torch.ones(1))
        self.b2 = nn.Parameter(torch.ones(1))

    def forward(self, x, A):
        # x: (batch, num_neurons, in_dim)
        neighbor_agg = torch.matmul(A, self.W_message(x)) # additive: sum_j(A_ij * message_j)
        z = self.W_mod(x)
        hadamard_agg = z * torch.matmul(A, z) # multiplicative: z_i (.) (A z)_i
        self_term = self.W_self(x)

        out = self.relu(self.norm(
            self.b * self_term + self.b1 * neighbor_agg + self.b2 * hadamard_agg
        )) # combine and activate

        return out # (batch, num_neurons, out_dim)
    
class GraphNeuralNetwork(nn.Module):
    def __init__(
        self,
        num_neurons,
        embedding_dim=64,
        hidden_dim=64,
        output_dim=1,
        num_layers=5,
    ):
        # embedding_dim: dimension of input embeedings per neuron (from SpikeEncoder)
        # hidden dim: dimension of hidden representation after message passing
        # output_dim: dimension of output per neuron

        super(GraphNeuralNetwork, self).__init__()

        # signed adjacency parameterized as two non-negative parts (exc - inh),
        self.learned_adj_exc = nn.Parameter(0.01 * torch.randn(num_neurons, num_neurons).abs())
        self.learned_adj_inh = nn.Parameter(0.01 * torch.randn(num_neurons, num_neurons).abs())

        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(GNNLayer(in_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

        # output layer (hidden_dim -> output_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim) # (batch, num_neurons, output_dim)

        self.A_scale = 1.0 # for ablation test
    
    def forward(self, embeddings):
        A_exc= torch.relu(self.learned_adj_exc)
        A_inh = torch.relu(self.learned_adj_inh)
        A_signed = A_exc - A_inh

        A = A_signed * (1 - torch.eye(A_signed.size(0), device=A_signed.device)) # zero the diagonal
        A = A * self.A_scale

        x = embeddings
        for layer in self.layers:
            x = layer(x, A)
        return self.fc_out(x).squeeze(-1)
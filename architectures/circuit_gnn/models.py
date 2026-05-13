import torch
import torch.nn as nn
import torch.nn.functional as F


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


# GNN layer variants

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


class SkipGNNLayer(GNNLayer):
    """GNNLayer with a residual shortcut to prevent oversmoothing."""

    def __init__(self, in_dim, out_dim, self_scale=0.5):
        super().__init__(in_dim, out_dim, self_scale)
        self.shortcut = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x, A):
        residual = self.shortcut(x)
        messages = self.W_message(x)
        self_term = self.W_self(x)
        neighbor_agg = torch.matmul(A, messages)
        out = self.relu(self.norm(neighbor_agg + self.self_scale * self_term)) + residual
        return out


class EdgeCondGNNLayer(nn.Module):
    """Message MLP receives the edge weight A[i,j] alongside the source embedding.

    For each directed edge (j → i), the message is W_message(cat(x_j, A[i,j])),
    then aggregated weighted by A[i,j].  Zero-weight edges contribute exactly zero.
    O(N^2) — suitable for small N typical in neural circuit data.
    """

    def __init__(self, in_dim, out_dim, self_scale=0.5):
        super().__init__()
        self.self_scale = self_scale
        mlp_dim = 2 * in_dim
        self.W_message = nn.Sequential(
            nn.Linear(in_dim + 1, mlp_dim, bias=True),
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
        # x: (batch, N, in_dim)   A: (N, N) where A[i,j] = j→i weight
        batch, N, _ = x.shape
        x_src = x.unsqueeze(1).expand(-1, N, -1, -1)                    # (batch, N_i, N_j, in_dim)
        A_exp = A.unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, 1)   # (batch, N_i, N_j, 1)
        edge_in = torch.cat([x_src, A_exp], dim=-1)                     # (batch, N, N, in_dim+1)
        edge_msg = self.W_message(edge_in)                               # (batch, N, N, out_dim)
        neighbor_agg = (A.unsqueeze(0).unsqueeze(-1) * edge_msg).sum(dim=2)  # (batch, N, out_dim)
        self_term = self.W_self(x)
        out = self.relu(self.norm(neighbor_agg + self.self_scale * self_term))
        return out


class SplitPathGNNLayer(nn.Module):
    """Separate message MLPs for excitatory (A > 0) and inhibitory (A < 0) connections.

    Allows the network to learn different transformations for inputs from
    excitatory vs inhibitory presynaptic neurons.
    """

    def __init__(self, in_dim, out_dim, self_scale=0.5):
        super().__init__()
        self.self_scale = self_scale
        mlp_dim = 2 * in_dim
        self.W_exc = nn.Sequential(
            nn.Linear(in_dim, mlp_dim, bias=True), nn.ReLU(),
            nn.Linear(mlp_dim, out_dim, bias=True),
        )
        self.W_inh = nn.Sequential(
            nn.Linear(in_dim, mlp_dim, bias=True), nn.ReLU(),
            nn.Linear(mlp_dim, out_dim, bias=True),
        )
        self.W_combine = nn.Linear(2 * out_dim, out_dim, bias=True)
        self.W_self = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True), nn.ReLU(),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        A_pos = F.relu(A)           # excitatory connections only
        A_neg = F.relu(-A)          # magnitude of inhibitory connections
        agg_exc = torch.matmul(A_pos, self.W_exc(x))
        agg_inh = torch.matmul(A_neg, self.W_inh(x))
        combined = self.W_combine(torch.cat([agg_exc, agg_inh], dim=-1))
        self_term = self.W_self(x)
        out = self.relu(self.norm(combined + self.self_scale * self_term))
        return out


# ---------------------------------------------------------------------------
# GNN models
# ---------------------------------------------------------------------------

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

        self.layers = nn.ModuleList(
            self._build_layers(num_layers, embedding_dim, hidden_dim, self_scale)
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.A_scale = 1.0

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(GNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers

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


class SkipGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork with residual skip connections in every GNN layer."""

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(SkipGNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers


class EdgeCondGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork where each message is conditioned on the edge weight A[i,j]."""

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(EdgeCondGNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers


class SplitPathGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork with separate message MLPs for exc and inh connections."""

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(SplitPathGNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers


class MultiStepGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork that predicts k future time steps simultaneously.

    fc_out produces (batch, N, k_steps) instead of (batch, N).
    Requires MultiStepSpikeTrainDataset which returns k-step targets.
    """

    def __init__(self, num_neurons, k_steps=4, **kwargs):
        kwargs = {**kwargs, 'output_dim': k_steps}  # fc_out → (hidden, k_steps)
        super().__init__(num_neurons, **kwargs)
        self.k_steps = k_steps

    def forward(self, embeddings):
        A = self.get_adjacency() * self.A_scale
        x = embeddings
        for layer in self.layers:
            x = layer(x, A)
        return self.fc_out(x)  # (batch, N, k_steps) — no squeeze


class DalesLawGNN(nn.Module):
    """Each presynaptic neuron is exclusively excitatory or inhibitory to all its targets.

    One NxN non-negative weight matrix + one Nx1 sign logit.
    A[i,j] = sign[j] * weights[i,j]  (sign on source/column axis).
    """

    def __init__(
        self,
        num_neurons,
        embedding_dim=64,
        hidden_dim=64,
        output_dim=1,
        num_layers=5,
        desired_max=0.05,
        init_noise=0.1,
        self_scale=0.5,
    ):
        super(DalesLawGNN, self).__init__()

        self._raw_weights = nn.Parameter(0.01 * torch.randn(num_neurons, num_neurons).abs())
        self._sign_logits = nn.Parameter(torch.zeros(num_neurons))

        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(GNNLayer(in_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.A_scale = 1.0

    @property
    def learned_adj_exc(self) -> torch.Tensor:
        weights = F.softplus(self._raw_weights)
        exc_scale = torch.clamp(torch.tanh(self._sign_logits), min=0)
        return weights * exc_scale.unsqueeze(0)  # column j scaled by max(sign[j], 0)

    @property
    def learned_adj_inh(self) -> torch.Tensor:
        weights = F.softplus(self._raw_weights)
        inh_scale = torch.clamp(-torch.tanh(self._sign_logits), min=0)
        return weights * inh_scale.unsqueeze(0)  # column j scaled by max(-sign[j], 0)

    def get_adjacency(self) -> torch.Tensor:
        weights = F.softplus(self._raw_weights)
        signs = torch.tanh(self._sign_logits)
        A = signs.unsqueeze(0) * weights            # col j = sign[j] * weights[:,j]
        return A * (1 - torch.eye(A.size(0), device=A.device))

    def forward(self, embeddings):
        A = self.get_adjacency() * self.A_scale
        x = embeddings
        for layer in self.layers:
            x = layer(x, A)
        return self.fc_out(x).squeeze(-1)


class IndirectFilterGNN(GraphNeuralNetwork):
    """Subtracts two-hop indirect effects from the adjacency before message passing.

    A_direct = A_raw - alpha * (A_raw @ A_raw), alpha learned in (0, 1].
    """

    def __init__(self, num_neurons, **kwargs):
        super(IndirectFilterGNN, self).__init__(num_neurons, **kwargs)
        self._log_alpha = nn.Parameter(torch.tensor(-2.0))

    def get_adjacency(self) -> torch.Tensor:
        A_raw = super().get_adjacency()
        alpha = torch.exp(self._log_alpha).clamp(max=1.0)
        A_two_hop = (A_raw @ A_raw).detach()
        A_direct = A_raw - alpha * A_two_hop
        return A_direct * (1 - torch.eye(A_direct.size(0), device=A_direct.device))

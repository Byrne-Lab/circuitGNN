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


# ===========================================================================
# Baseline: standard message-passing layer + base GNN model
# Every variant below pairs its custom layer (if any) with its model class.
# Models subclass GraphNeuralNetwork and typically override only _build_layers.
# ===========================================================================

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

        self._init_adjacency(num_neurons, A_init, signed_init, desired_max, init_noise)
        self.layers = nn.ModuleList(
            self._build_layers(num_layers, embedding_dim, hidden_dim, self_scale)
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.A_scale = 1.0

    def _init_adjacency(self, num_neurons, A_init, signed_init, desired_max, init_noise):
        """Register the learnable adjacency parameters. Subclasses with a
        different connectivity parameterization (e.g. DalesLawGNN) override this."""
        if A_init is not None and not signed_init:
            print("Using unsigned adjacency initialization.")
            # use |A_init| magnitudes with independent noise per matrix, floored at a
            # small positive value: relu'(x)=0 for x<=0, so non-positive raw entries
            # would never receive gradient and the edge would be permanently dead
            A_max = A_init.abs().max()
            A_norm = (A_init.abs() / A_max) * desired_max
            init_exc = (A_norm + torch.randn_like(A_norm) * init_noise).clamp_min(1e-4)
            init_inh = (A_norm + torch.randn_like(A_norm) * init_noise).clamp_min(1e-4)
            self.learned_adj_exc = nn.Parameter(init_exc)
            self.learned_adj_inh = nn.Parameter(init_inh)

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
            # floor all entries at a small positive value so edges outside the
            # prior's sign masks stay trainable (relu gradient is zero at 0)
            self.learned_adj_exc = nn.Parameter(init_exc.clamp_min(1e-4))
            self.learned_adj_inh = nn.Parameter(init_inh.clamp_min(1e-4))

        else:
            print("Using random numbers for adjacency initialization.")
            self.learned_adj_exc = nn.Parameter(0.01 * torch.randn(num_neurons, num_neurons).abs())
            self.learned_adj_inh = nn.Parameter(0.01 * torch.randn(num_neurons, num_neurons).abs())

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


# ===========================================================================
# Dales' law: each presynaptic neuron is exclusively excitatory or inhibitory
# ===========================================================================

class DalesLawGNN(GraphNeuralNetwork):
    """Each presynaptic neuron is exclusively excitatory or inhibitory to all its targets.

    One NxN non-negative weight matrix + one Nx1 sign logit.
    A[i,j] = sign[j] * weights[i,j]  (sign on source/column axis).

    Reuses GraphNeuralNetwork's layers/forward; only the adjacency
    parameterization differs, so it overrides _init_adjacency + get_adjacency
    and exposes learned_adj_exc/inh as properties (for the training penalties).
    """

    def _init_adjacency(self, num_neurons, A_init, signed_init, desired_max, init_noise):
        # A_init / signed_init are ignored: Dales' law fixes the sign structure.
        # softplus(0.01·|randn|) ≈ 0.70, ~14x desired_max — initialize raw weights at
        # softplus^{-1}(desired_max) so starting magnitudes match the other variants
        raw_init = torch.log(torch.expm1(torch.tensor(float(desired_max))))
        self._raw_weights = nn.Parameter(
            raw_init + init_noise * torch.randn(num_neurons, num_neurons)
        )
        # tanh(0)=0 zeroes the adjacency AND the weight gradients (dA/dw ∝ sign);
        # small noise breaks the symmetry so both sets of params train from epoch 1
        self._sign_logits = nn.Parameter(0.01 * torch.randn(num_neurons))

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


# ===========================================================================
# Indirect filter: subtract two-hop effects from the adjacency
# ===========================================================================

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


# ===========================================================================
# Skip / residual connections
# ===========================================================================

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


class SkipGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork with residual skip connections in every GNN layer."""

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(SkipGNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers


# ===========================================================================
# Edge-conditioned messages: W_message sees the edge weight A[i,j]
# ===========================================================================

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


class EdgeCondGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork where each message is conditioned on the edge weight A[i,j]."""

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(EdgeCondGNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers


# ===========================================================================
# Split path: separate message MLPs for excitatory / inhibitory connections
# ===========================================================================

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


class SplitPathGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork with separate message MLPs for exc and inh connections."""

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(SplitPathGNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers


# ===========================================================================
# Multi-step: predict k future time steps simultaneously (reuses GNNLayer)
# ===========================================================================

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


# ===========================================================================
# Learned scale: learnable scalars b1, b2 on neighbor and self terms
# ===========================================================================

class LearnedScaleGNNLayer(nn.Module):
    """GNNLayer where neighbor and self contributions have learnable scalar weights b1, b2."""

    def __init__(self, in_dim, out_dim, self_scale=0.5):
        super().__init__()
        mlp_dim = 2 * in_dim
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
        self.b1 = nn.Parameter(torch.ones(1))
        self.b2 = nn.Parameter(torch.tensor(float(self_scale)))

    def forward(self, x, A):
        neighbor_agg = torch.matmul(A, self.W_message(x))
        self_term = self.W_self(x)
        out = self.relu(self.norm(self.b1 * neighbor_agg + self.b2 * self_term))
        return out


class LearnedScaleGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork with learnable b1/b2 scalars on neighbor and self terms."""

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(LearnedScaleGNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers


# ===========================================================================
# Hadamard modulator: additive + multiplicative message passing (AMAG-inspired)
# ===========================================================================

class HadamardModGNNLayer(nn.Module):
    """GNN layer with additive + Hadamard-modulator message passing.

    Inspired by AMAG (Li et al. NeurIPS 2023):
      z_i = b*W_self(x_i) + b1*(A @ W_msg(x))_i + b2*W_mod(x_i ⊙ (A @ x)_i)

    The Hadamard term A_ij*(x_i⊙x_j) summed over j equals x_i ⊙ (A @ x)_i.
    Both paths share the same adjacency matrix. b, b1, b2 are learnable scalars.
    """

    def __init__(self, in_dim, out_dim, self_scale=0.5):
        super().__init__()
        mlp_dim = 2 * in_dim
        self.W_message = nn.Sequential(
            nn.Linear(in_dim, mlp_dim, bias=True),
            nn.ReLU(),
            nn.Linear(mlp_dim, out_dim, bias=True),
        )
        self.W_mod = nn.Linear(in_dim, out_dim, bias=True)
        self.W_self = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.b  = nn.Parameter(torch.tensor(float(self_scale)))
        self.b1 = nn.Parameter(torch.ones(1))
        self.b2 = nn.Parameter(torch.ones(1))

    def forward(self, x, A):
        neighbor_agg = torch.matmul(A, self.W_message(x))  # (batch, N, out_dim)
        hadamard_agg = self.W_mod(x * torch.matmul(A, x))  # (batch, N, out_dim)
        self_term    = self.W_self(x)                       # (batch, N, out_dim)
        out = self.relu(self.norm(
            self.b * self_term + self.b1 * neighbor_agg + self.b2 * hadamard_agg
        ))
        return out


class HadamardModGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork using HadamardModGNNLayer in every message-passing step."""

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(HadamardModGNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers


# ===========================================================================
# Diff: messages from pairwise embedding differences h_i - h_j
# ===========================================================================

class DiffGNNLayer(nn.Module):
    """Message from j→i is W_message(h_i - h_j), gated by A[i,j], then aggregated.

    diff[i,j] = h_i - h_j for all pairs, passed through W_message, then weighted by A.
    O(N^2) — suitable for small N typical in neural circuit data.
    """

    def __init__(self, in_dim, out_dim, self_scale=0.5):
        super().__init__()
        self.self_scale = self_scale
        mlp_dim = 2 * in_dim
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
        # x: (batch, N, in_dim)   A: (N, N) where A[i,j] = j→i weight
        _, N, _ = x.shape
        x_i = x.unsqueeze(2).expand(-1, -1, N, -1)   # (batch, N_i, N_j, in_dim)
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)   # (batch, N_i, N_j, in_dim)
        diff = x_i - x_j                               # (batch, N_i, N_j, in_dim)
        edge_msg = self.W_message(diff)                # (batch, N_i, N_j, out_dim)
        neighbor_agg = (A.unsqueeze(0).unsqueeze(-1) * edge_msg).sum(dim=2)  # (batch, N, out_dim)
        self_term = self.W_self(x)
        out = self.relu(self.norm(neighbor_agg + self.self_scale * self_term))
        return out


class DiffGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork where messages are W_message(h_i - h_j) gated by A[i,j]."""

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(DiffGNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers


# ===========================================================================
# Softmax: transformer-style row-wise attention over incoming edges
# ===========================================================================

class SoftmaxGNN(GraphNeuralNetwork):
    """Transformer-style attention: row-wise softmax over incoming edge weights.

    The signed adjacency entries act as attention logits; each neuron's incoming
    weights are normalized to sum to 1 via softmax (diagonal masked out) with a
    learnable temperature. A_scale multiplies the normalized matrix so the
    A_scale=0 ablation still removes all neighbor information.

    get_adjacency() still returns the raw signed logits, which remain the
    interpretable connectivity estimate (cf. NetFormer's argument that softmax
    constrains in-degrees to sum to 1 — this variant tests that constraint).
    """

    def __init__(self, num_neurons, **kwargs):
        super().__init__(num_neurons, **kwargs)
        self._log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, embeddings):
        A = self.get_adjacency()
        temp = torch.exp(self._log_temp)
        eye = torch.eye(A.size(0), dtype=torch.bool, device=A.device)
        logits = (A / temp).masked_fill(eye, float('-inf'))
        A_eff = torch.softmax(logits, dim=-1) * self.A_scale
        x = embeddings
        for layer in self.layers:
            x = layer(x, A_eff)
        return self.fc_out(x).squeeze(-1)


# ===========================================================================
# Per-neuron self scale: each neuron learns its own self-term weight
# ===========================================================================

class PerNeuronSelfScaleGNNLayer(nn.Module):
    """GNNLayer with a learnable per-neuron self-term scale vector.

    Each neuron i gets its own scalar s_i weighting its self contribution:
    out_i = relu(norm(neighbor_agg_i + s_i * self_term_i)).
    """

    def __init__(self, in_dim, out_dim, num_neurons, self_scale=0.5):
        super().__init__()
        mlp_dim = 2 * in_dim
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
        self.self_scales = nn.Parameter(torch.full((num_neurons,), float(self_scale)))

    def forward(self, x, A):
        neighbor_agg = torch.matmul(A, self.W_message(x))
        self_term = self.W_self(x)
        out = self.relu(self.norm(neighbor_agg + self.self_scales.view(1, -1, 1) * self_term))
        return out


class PerNeuronSelfScaleGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork where each neuron's self-term scale is a learned scalar."""

    def __init__(self, num_neurons, **kwargs):
        self._num_neurons = num_neurons
        super().__init__(num_neurons, **kwargs)

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(PerNeuronSelfScaleGNNLayer(
                in_dim, hidden_dim, num_neurons=self._num_neurons, self_scale=self_scale))
        return layers


# ===========================================================================
# Gated: aggregated messages update node state through a GRU cell (Yoon-inspired)
# ===========================================================================

class GatedGNNLayer(nn.Module):
    """Aggregated neighbor messages update the node state through a GRU cell.

    Inspired by the spike prediction module of Yoon (NeurIPS 2025):
    h_i' = GRU(sum_j A[i,j] * msg_j, h_i). The GRU's gates learn how much of
    the self state to keep vs how much neighbor input to integrate, replacing
    the fixed self_scale mixing (self_scale is accepted but unused).
    """

    def __init__(self, in_dim, out_dim, self_scale=0.5):
        super().__init__()
        mlp_dim = 2 * in_dim
        self.W_message = nn.Sequential(
            nn.Linear(in_dim, mlp_dim, bias=True),
            nn.ReLU(),
            nn.Linear(mlp_dim, out_dim, bias=True),
        )
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.gru = nn.GRUCell(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, A):
        batch, N, _ = x.shape
        msg = torch.matmul(A, self.W_message(x))            # (batch, N, out_dim)
        h = self.proj(x)                                     # (batch, N, out_dim)
        out = self.gru(msg.reshape(batch * N, -1), h.reshape(batch * N, -1))
        return self.norm(out.view(batch, N, -1))


class GatedGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork where each layer integrates messages via a GRU cell."""

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(GatedGNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers


# ===========================================================================
# Adaptive: AMAG-style sample-dependent Adaptor modulating a shared adjacency
# ===========================================================================

class AdaptiveGNNLayer(nn.Module):
    """AMAG-style sample-dependent Adaptor modulating a shared adjacency.

    S[b,i,j] = sigmoid(MLP([x_i, x_j])) in (0, 1) rescales each edge per input
    sample: agg_i = sum_j S[b,i,j] * A[i,j] * msg_j  (Li et al. NeurIPS 2023, Eq. 3).
    The static A captures average connectivity; S captures state-dependent gain.
    O(N^2) — suitable for small N typical in neural circuit data.
    """

    def __init__(self, in_dim, out_dim, self_scale=0.5):
        super().__init__()
        self.self_scale = self_scale
        mlp_dim = 2 * in_dim
        self.W_message = nn.Sequential(
            nn.Linear(in_dim, mlp_dim, bias=True),
            nn.ReLU(),
            nn.Linear(mlp_dim, out_dim, bias=True),
        )
        self.adaptor = nn.Sequential(
            nn.Linear(2 * in_dim, in_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim, 1, bias=True),
        )
        self.W_self = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        # x: (batch, N, in_dim)   A: (N, N) where A[i,j] = j→i weight
        _, N, _ = x.shape
        x_i = x.unsqueeze(2).expand(-1, -1, N, -1)   # (batch, N_i, N_j, in_dim) target
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)   # (batch, N_i, N_j, in_dim) source
        S = torch.sigmoid(self.adaptor(torch.cat([x_i, x_j], dim=-1))).squeeze(-1)  # (batch, N, N)
        A_adapted = S * A.unsqueeze(0)                                # (batch, N, N)
        neighbor_agg = torch.bmm(A_adapted, self.W_message(x))       # (batch, N, out_dim)
        self_term = self.W_self(x)
        out = self.relu(self.norm(neighbor_agg + self.self_scale * self_term))
        return out


class AdaptiveGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork with an AMAG-style sample-dependent edge Adaptor."""

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(AdaptiveGNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers


# ===========================================================================
# ReLU message: rectify the message path so it's symmetric with W_self
# ===========================================================================

class ReLUMessageGNNLayer(GNNLayer):
    """Baseline GNNLayer with a ReLU on the message path too, so both pathways
    are symmetric (non-negative pre-mix).

    In the baseline, W_message ends in a Linear (signed output) while W_self ends
    in a ReLU (non-negative output). This variant rectifies the message output as
    well: messages become non-negative per feature, so the *sign* of a neighbor's
    influence is carried entirely by the signed edge weight A[i,j] rather than
    being shared between the edge and the message content. Tests whether that
    asymmetry helped or hurt.
    """

    def __init__(self, in_dim, out_dim, self_scale=0.5):
        super().__init__(in_dim, out_dim, self_scale)
        mlp_dim = 2 * in_dim
        self.W_message = nn.Sequential(
            nn.Linear(in_dim, mlp_dim, bias=True),
            nn.ReLU(),
            nn.Linear(mlp_dim, out_dim, bias=True),
            nn.ReLU(),
        )


class ReLUMessageGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork whose message path is rectified, matching W_self."""

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(ReLUMessageGNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers


# ===========================================================================
# Autoregressive residual: condition on each neuron's own history first, then
# let the graph explain only the residual (conditional-Granger inductive bias)
# ===========================================================================

class ARResidualGNN(GraphNeuralNetwork):
    """Autoregressive-residual model that conditions on a neuron's own history first.

    A per-neuron self head predicts the next-bin logit from that neuron's OWN
    embedding alone. The graph message-passing path (inherited layers + fc_out)
    predicts only the residual the self head cannot explain:

        logit_i = self_pred_i.detach() + residual_i

    The detach routes the main loss entirely onto the residual path (it must fit
    target - self_pred), while an auxiliary loss (added in compute_loss when
    ``last_self_pred`` is exposed) trains the self head to be a strong standalone
    predictor. Consequence: a downstream neuron that is merely a delayed copy of
    its source carries no information about the source beyond the source's own
    history, so the residual gives it no gradient and the phantom reverse edge
    never forms — while a genuine source (which adds predictive info beyond the
    target's own past) still earns its forward edge.

    Only the parameterization of the prediction differs; get_adjacency() and the
    connectivity parameters are inherited unchanged, so the same penalties,
    ablation and extraction all work as for the baseline.
    """

    def __init__(self, num_neurons, self_aux_lambda=1.0, **kwargs):
        super().__init__(num_neurons, **kwargs)
        embedding_dim = kwargs.get('embedding_dim', 64)
        hidden_dim = kwargs.get('hidden_dim', 64)
        output_dim = kwargs.get('output_dim', 1)
        self.self_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.self_aux_lambda = self_aux_lambda
        # stashed each forward so compute_loss can add the auxiliary self-loss
        self.last_self_pred = None

    def forward(self, embeddings):
        # self head: predict the target from each neuron's OWN embedding only
        self_pred = self.self_head(embeddings).squeeze(-1)   # (batch, N)
        self.last_self_pred = self_pred

        # graph path predicts the residual the self head cannot explain
        A = self.get_adjacency() * self.A_scale
        x = embeddings
        for layer in self.layers:
            x = layer(x, A)
        residual = self.fc_out(x).squeeze(-1)                # (batch, N)

        # detach: the main loss trains the residual path against (target - self_pred);
        # the self head is trained solely by the auxiliary loss in compute_loss
        return self_pred.detach() + residual

import torch
import torch.nn as nn
import torch.nn.functional as F


# ARCHITECTURE VARIANTS — INDEX
#
# Shared scaffold: SpikeEncoder turns a spike window into per-neuron embeddings;
# GraphNeuralNetwork learns a signed N x N adjacency (get_adjacency) and runs
# `num_layers` message-passing steps to predict each neuron's next bin. Every
# variant changes ONE axis of this scaffold and is selected by the `model_type`
# string in train.build_models. The ENCODER is a separate axis selected by
# `encoder_type` (default "standard"; "memory_slot" adds a per-neuron memory).
#
#   #  model_type        one-line idea
#   1  standard          Baseline: signed scalar adjacency + 2-layer message MLP.
#   2  hadamard_mod      Add a multiplicative x_i (.) (A x)_i path (AMAG-style).
#   3  multi_step        Predict k future bins at once (forces propagation).
#   4  AGGREGATION axis — how neighbour messages are reduced over the learned A:
#      agg_sum           sum_j A[i,j] m_j                          (== baseline).
#      agg_mean          sum_j A[i,j] m_j / sum_j |A[i,j]|.
#      agg_max           elementwise max_j ( A[i,j] m_j ).
#      agg_vpa           sum_j A[i,j] m_j / sqrt(sum_j A[i,j]^2)   (variance-pres.)
#      agg_attention     GAT attention REPLACES A; readout = mean attention.
#   5  UPDATE axis — how the aggregated message integrates into the node state:
#      update_gru        GRU cell mixes neighbour message into node state.
#      update_conv       Conv1D over the feature dim mixes self + message.
#   6  interpretable     Baseline (identical training) + analysis probes that
#                        read effective connectivity out of message passing:
#                        Jacobian connectivity & mono-vs-polysynaptic split.
#
# Aggregation refs: GNN-VPA (Schneckenreiter et al. 2024) for sum/mean/max/vpa;
# GAT (Velickovic et al. 2018) for attention. Update ref: GGS-NN (Li et al. 2016)
# for the GRU cell.


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
        x = F.pad(x, (self.kernel_size - 1, 0))  # left-only causal pad
        out = self.conv1d(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = out.view(batch_size * num_neurons, -1)
        embeddings = self.fc(out)
        embeddings = embeddings.view(batch_size, num_neurons, -1)
        return embeddings


# ===========================================================================
# ENCODER VARIANT — Per-neuron memory                (encoder_type="memory_slot")
# Idea: give each neuron a persistent, learnable memory vector M[i] that is
# concatenated into its window encoding before the final projection. The conv
# path still summarizes the current spike window; the memory slot adds a stable
# per-neuron "identity / baseline" the window alone cannot express (a learnable
# per-neuron prior). One slot row per neuron, shared across all windows; updated
# only by gradient descent. Selected by `encoder_type` in train.build_models.
#
# Note: this ties the model to a fixed neuron ordering (slot i belongs to neuron
# i), which is fine for these fixed circuits but is not transferable across
# datasets with different neurons. It needs num_neurons at construction.
# ===========================================================================

class MemorySpikeEncoder(nn.Module):
    """SpikeEncoder with a persistent learnable memory vector per neuron.

    Identical conv/window path to SpikeEncoder, but a (num_neurons, mem_dim)
    parameter table is concatenated to each neuron's flattened conv features
    before the fc projection, so neuron i always carries its own memory row.
    """

    def __init__(self, window_size, num_neurons, conv_out_channels=32,
                 embedding_dim=64, kernel_size=3, mem_dim=16):
        super(MemorySpikeEncoder, self).__init__()

        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(in_channels=1,
                                out_channels=conv_out_channels,
                                kernel_size=kernel_size,
                                padding=0)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(conv_out_channels)
        # one persistent memory row per neuron (breaks permutation symmetry)
        self.neuron_memory = nn.Parameter(0.01 * torch.randn(num_neurons, mem_dim))
        self.fc = nn.Linear(conv_out_channels * window_size + mem_dim, embedding_dim)

    def forward(self, x):
        batch_size, num_neurons, window_size = x.shape
        x = x.view(batch_size * num_neurons, 1, window_size)
        x = F.pad(x, (self.kernel_size - 1, 0))  # left-only causal pad
        out = self.conv1d(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = out.view(batch_size, num_neurons, -1)
        # append each neuron's persistent memory row, broadcast over the batch
        mem = self.neuron_memory.unsqueeze(0).expand(batch_size, -1, -1)
        out = torch.cat([out, mem], dim=-1)
        embeddings = self.fc(out)
        return embeddings


# ===========================================================================
# VARIANT 1 — Baseline                                  (model_type="standard")
# Idea: encode each neuron's spike window, learn a signed N x N adjacency, and
# message-pass `num_layers` times to predict the next bin. Every other variant
# is a single-axis tweak of this scaffold (a custom layer paired with its model
# class, overriding only _build_layers unless noted).
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
        neighbor_agg = torch.matmul(A, messages)  # sum_j A[i,j] * message_j
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
        self.A_scale = 1.0  # for the neighbour-ablation test (set to 0 to remove A)

    def _init_adjacency(self, num_neurons, A_init, signed_init, desired_max, init_noise):
        """Register the learnable adjacency parameters. Variants with a different
        connectivity parameterization override this."""
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
# VARIANT 6 — Interpretability probe               (model_type="interpretable")
# Idea: behaviourally identical to the baseline (same GNNLayer scaffold, same
# learned signed A from _init_adjacency, same forward at train time, same
# regularizers) but adds analysis methods that read interpretability signals
# OUT of the trained message passing. The point is that get_adjacency() returns
# the raw weight A[i,j], whereas each message actually passes through W_message
# (a ReLU MLP) + LayerNorm, so the influence a source truly exerts on a target
# is not A but the input->output sensitivity of the network. These probes
# measure that, and split it by synaptic path length.
#
#   • jacobian_connectivity(emb, n_hops)     — effective connectivity J[i,j]:
#       how much source j moves target i's predicted logit, as the Jacobian of
#       the output w.r.t. the per-neuron input embedding (reduced over features).
#   • polysynaptic_decomposition(emb)        — the same influence split by path
#       length, by limiting how many message-passing layers may use A:
#       n_hops=1 -> monosynaptic (direct), up to L -> full polysynaptic reach.
#   • module fn signed_input_jacobian(enc,gnn,x) — signed effective connectivity
#       through the FULL pipeline (encoder+GNN) for an excitatory/inhibitory sign
#       directly comparable to the signed get_adjacency().
#
# forward(emb, n_hops=None) is the ONLY change vs baseline, and n_hops=None
# reproduces GraphNeuralNetwork.forward exactly, so training is unchanged and
# extract_adjacency()/get_adjacency() behave like the baseline.
# ===========================================================================

class InterpretableGNN(GraphNeuralNetwork):
    """Baseline GNN plus effective-connectivity probes (see banner)."""

    def forward(self, embeddings, n_hops=None):
        """Baseline forward, optionally restricting neighbour mixing to the first
        `n_hops` message-passing layers (later layers update self-only, A=0).

        The longest source->target path then spans `n_hops` edges, so n_hops=1 is
        monosynaptic and n_hops=L is the full polysynaptic reach. n_hops=None uses
        A in every layer -> identical to GraphNeuralNetwork.forward (train path).
        """
        A = self.get_adjacency() * self.A_scale
        Z = torch.zeros_like(A)
        x = embeddings
        for li, layer in enumerate(self.layers):
            x = layer(x, A if (n_hops is None or li < n_hops) else Z)
        return self.fc_out(x).squeeze(-1)

    def jacobian_connectivity(self, embeddings, n_hops=None, signed=False):
        """Effective connectivity J[i,j] = influence of source j on target i.

        Computes the Jacobian of the output logits w.r.t. the per-neuron input
        embeddings and reduces over the embedding feature dim, averaged over the
        batch. Orientation is [post, pre] = [target i, source j], matching
        get_adjacency() (the runner transposes once at save time).

            signed=False : ||d y_i / d emb_j||_2            (magnitude, vs |A|)
            signed=True  : magnitude * sign(sum_d d y_i/d emb_j)   (heuristic
                           sign; for a faithful E/I sign use signed_input_jacobian
                           through the encoder, where "more input spikes" has a
                           well-defined direction).

        Put the model in eval() first if you want deterministic BatchNorm/Dropout
        upstream. Returns an (N, N) CPU tensor.
        """
        emb = embeddings.detach().clone().requires_grad_(True)
        out = self.forward(emb, n_hops=n_hops)               # (B, N) logits
        N = out.shape[1]
        J = torch.zeros(N, N, device=emb.device)
        for i in range(N):
            g, = torch.autograd.grad(out[:, i].sum(), emb, retain_graph=(i < N - 1))
            mag = g.norm(dim=-1)                              # (B, N_source)
            if signed:
                mag = mag * torch.sign(g.sum(dim=-1))
            J[i] = mag.mean(dim=0)
        return J.detach().cpu()

    def polysynaptic_decomposition(self, embeddings, max_hops=None, signed=False):
        """Split effective connectivity by synaptic path length.

        Runs jacobian_connectivity at n_hops = 1..max_hops (default: number of
        message-passing layers) and returns:
            'cumulative'[k-1]  : influence using up to k hops
            'incremental'[k-1] : influence of EXACTLY-k-hop paths (J_k - J_{k-1})
            'mono'             : incremental[0] (direct / monosynaptic)
            'poly'             : cumulative[-1] - cumulative[0] (all indirect)
        Comparing 'mono' against get_adjacency() shows how much of a learned edge
        is genuinely direct vs a shadow of multi-step propagation (the confound
        pairwise correlation / transfer entropy collapse into one number).
        """
        L = len(self.layers) if max_hops is None else max_hops
        cum = [self.jacobian_connectivity(embeddings, n_hops=k, signed=signed)
               for k in range(1, L + 1)]
        inc = [cum[0]] + [cum[k] - cum[k - 1] for k in range(1, L)]
        return {"cumulative": cum, "incremental": inc,
                "mono": cum[0], "poly": cum[-1] - cum[0]}


def signed_input_jacobian(encoder, gnn, inputs, n_hops=None):
    """Signed effective connectivity through the FULL pipeline (encoder + GNN).

    J[i,j] = mean_batch sum_t  d y_i / d x_{j,t}: if source neuron j spikes more
    across its input window, does target i's predicted logit rise (excitatory, +)
    or fall (inhibitory, -)? Summing the gradient over the window collapses it to
    a signed scalar per source, directly comparable to the signed get_adjacency().

    inputs: (B, N, W) raw spike windows. `gnn` should be an InterpretableGNN if
    n_hops is not None (so mono/poly paths can be isolated); for n_hops=None any
    baseline GNN works. Orientation is [post, pre] = [target i, source j].
    """
    encoder.eval(); gnn.eval()
    x = inputs.detach().clone().requires_grad_(True)
    emb = encoder(x)
    out = gnn(emb, n_hops=n_hops) if n_hops is not None else gnn(emb)
    N = out.shape[1]
    J = torch.zeros(N, N, device=x.device)
    for i in range(N):
        g, = torch.autograd.grad(out[:, i].sum(), x, retain_graph=(i < N - 1))
        J[i] = g.sum(dim=-1).mean(dim=0)                     # signed, per source
    return J.detach().cpu()


# ===========================================================================
# VARIANT 2 — Hadamard modulator                    (model_type="hadamard_mod")
# Idea: add a multiplicative message path x_i (.) (A x)_i alongside the additive
# one, with learnable mixing scalars (AMAG-style). Captures gain / coincidence
# interactions that a purely additive aggregator cannot represent.
# ===========================================================================

class HadamardModGNNLayer(nn.Module):
    """GNN layer with additive + Hadamard-modulator message passing.

    Inspired by AMAG (Li et al. NeurIPS 2023):
      z   = W_mod(x)
      out = b*W_self(x_i) + b1*(A @ W_msg(x))_i + b2*(z_i ⊙ (A @ z)_i)

    The Hadamard modulation runs in the projected space: the term
    z_i ⊙ (A @ z)_i equals sum_j A_ij*(z_i ⊙ z_j), so each node is modulated
    by its (presynaptic) neighbors after both have passed through W_mod.
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
        self.b  = nn.Parameter(torch.tensor(float(self_scale)))
        self.b1 = nn.Parameter(torch.ones(1))
        self.b2 = nn.Parameter(torch.ones(1))

    def forward(self, x, A):
        neighbor_agg = torch.matmul(A, self.W_message(x))  # (batch, N, out_dim)
        z            = self.W_mod(x)                        # (batch, N, out_dim)
        hadamard_agg = z * torch.matmul(A, z)               # (batch, N, out_dim)
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
# VARIANT 3 — Multi-step                               (model_type="multi_step")
# Idea: predict k future bins simultaneously instead of just t+1. Forces the
# adjacency to capture multi-step propagation rather than one-step correlation.
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
# VARIANT 4 — Aggregation functions
#   (model_type="agg_sum" | "agg_mean" | "agg_max" | "agg_vpa")
# Idea: keep the baseline message MLP, self term, norm and update, but change
# how neighbour messages are reduced over the (signed, learned, weighted)
# adjacency A. The per-edge contribution is A[i,j]*m_j; the aggregator only sets
# how that is combined over the sources j (GNN-VPA, Schneckenreiter et al. 2024):
#
#     sum  : sum_j A[i,j] m_j                          (== baseline)
#     mean : sum_j A[i,j] m_j / sum_j |A[i,j]|
#     max  : elementwise  max_j ( A[i,j] m_j )
#     vpa  : sum_j A[i,j] m_j / sqrt(sum_j A[i,j]^2)
#
# A stays the connectivity readout (get_adjacency) and keeps full gradient in
# every mode. The mean/vpa denominators are per-target-row scalars with an eps
# so the A_scale=0 ablation (A==0) yields a zero neighbour term, not NaN.
# ===========================================================================

class AggregationGNNLayer(nn.Module):
    """Baseline layer with a selectable neighbour reduction (see banner)."""

    def __init__(self, in_dim, out_dim, self_scale=0.5, agg="sum", eps=1e-6):
        super().__init__()
        assert agg in ("sum", "mean", "max", "vpa"), f"unknown agg {agg!r}"
        self.self_scale = self_scale
        self.agg = agg
        self.eps = eps
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
        messages = self.W_message(x)                                  # (B, N, out)
        if self.agg == "max":
            # per-feature max over sources j of A[i,j] * m_j
            weighted = A.unsqueeze(0).unsqueeze(-1) * messages.unsqueeze(1)  # (B, N_i, N_j, out)
            neighbor_agg = weighted.max(dim=2).values                  # (B, N_i, out)
        else:
            neighbor_agg = torch.matmul(A, messages)                   # sum_j A[i,j] m_j
            if self.agg == "mean":
                denom = A.abs().sum(dim=1).view(1, -1, 1) + self.eps    # (1, N, 1)
                neighbor_agg = neighbor_agg / denom
            elif self.agg == "vpa":
                denom = torch.sqrt((A ** 2).sum(dim=1)).view(1, -1, 1) + self.eps
                neighbor_agg = neighbor_agg / denom
            # "sum" leaves neighbor_agg unchanged
        self_term = self.W_self(x)
        return self.relu(self.norm(neighbor_agg + self.self_scale * self_term))


class AggregationGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork with a configurable neighbour-aggregation operator."""

    def __init__(self, num_neurons, agg="sum", **kwargs):
        self._agg = agg
        super().__init__(num_neurons, **kwargs)

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(AggregationGNNLayer(in_dim, hidden_dim, self_scale=self_scale, agg=self._agg))
        return layers


# ===========================================================================
# VARIANT 4e — Attention aggregator                  (model_type="agg_attention")
# Idea: GAT-style attention (Velickovic et al. 2018) REPLACES the learned
# adjacency in the neighbour aggregation. There is no learned A; the connectivity
# readout is the running, sample-averaged layer-0 attention matrix.
# ===========================================================================

class GATLayer(nn.Module):
    """Single-head graph-attention layer, adapted to the scaffold's self +
    neighbour split. Attention is computed from node features and is the only
    neighbour weighting (the learned adjacency is not used):

        e_ij  = LeakyReLU( a_src . W h_i + a_dst . W h_j )
        alpha = softmax_j(e_ij)             (diagonal masked; i attends to others)
        agg_i = sum_j alpha_ij (W h_j)
        out_i = relu(norm( W_self(h_i) + a_scale * agg_i ))

    a_scale gates the neighbour term so the A_scale=0 ablation reduces the layer
    to self-only. The (batch-mean) attention is returned so the model can
    accumulate a sample-averaged attention matrix as its connectivity readout.
    """

    def __init__(self, in_dim, out_dim, self_scale=0.5, leaky_slope=0.2):
        super().__init__()
        self.self_scale = self_scale
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)   # a . W h_i  (target i)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)   # a . W h_j  (source j)
        self.leaky = nn.LeakyReLU(leaky_slope)
        self.W_self = nn.Sequential(nn.Linear(in_dim, out_dim, bias=True), nn.ReLU())
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, a_scale=1.0):
        # x: (B, N, in)
        _, N, _ = x.shape
        Wh = self.W(x)                                   # (B, N, out)
        e = self.leaky(self.a_src(Wh) + self.a_dst(Wh).transpose(1, 2))  # (B, N_i, N_j)
        eye = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
        e = e.masked_fill(eye, float('-inf'))
        alpha = torch.softmax(e, dim=-1)                 # (B, N_i, N_j) over sources j
        agg = torch.bmm(alpha, Wh)                       # (B, N, out): sum_j alpha_ij Wh_j
        out = self.relu(self.norm(self.W_self(x) + a_scale * agg))
        return out, alpha


class AttentionGNN(GraphNeuralNetwork):
    """GAT model where attention replaces the learned adjacency.

    There is no learned A. The connectivity readout (get_adjacency) is the
    running, sample-averaged layer-0 attention matrix accumulated over forward
    passes. To extract from the trained model, call reset_attention() and run a
    pass over data so the buffer reflects the current weights, then read
    get_adjacency(). attn[i,j] = weight of source j on target i ([post, pre]),
    matching the orientation of the baseline's A.
    """

    def __init__(self, num_neurons, leaky_slope=0.2, **kwargs):
        self._leaky_slope = leaky_slope
        super().__init__(num_neurons, **kwargs)
        # non-persistent: kept out of the checkpoint; repopulated at extraction
        self.register_buffer("_attn_sum", torch.zeros(num_neurons, num_neurons), persistent=False)
        self.register_buffer("_attn_count", torch.zeros(1), persistent=False)

    def _init_adjacency(self, num_neurons, A_init, signed_init, desired_max, init_noise):
        # no learned adjacency: attention replaces A, so there is no
        # learned_adj_exc/inh and compute_loss's adjacency penalties are skipped
        self.num_neurons = num_neurons

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(GATLayer(in_dim, hidden_dim, self_scale=self_scale,
                                   leaky_slope=self._leaky_slope))
        return layers

    def reset_attention(self):
        self._attn_sum.zero_()
        self._attn_count.zero_()

    def forward(self, embeddings):
        x = embeddings
        attn0 = None
        for li, layer in enumerate(self.layers):
            x, alpha = layer(x, a_scale=self.A_scale)
            if li == 0:
                attn0 = alpha
        # accumulate batch-mean layer-0 attention as the connectivity readout
        self._attn_sum += attn0.mean(dim=0).detach()
        self._attn_count += 1.0
        return self.fc_out(x).squeeze(-1)

    def get_adjacency(self) -> torch.Tensor:
        if float(self._attn_count) == 0:
            return torch.zeros_like(self._attn_sum)
        A = self._attn_sum / self._attn_count
        return A * (1 - torch.eye(A.size(0), device=A.device))


# ===========================================================================
# VARIANT 5a — GRU update                                   (model_type="update_gru")
# Idea: integrate the aggregated neighbor message into the node state with a GRU
# cell instead of a fixed additive mix. The gates learn how much self-state to
# keep vs neighbor input to absorb (the GRU runs over GNN depth, not time).
# ===========================================================================

class GatedGNNLayer(nn.Module):
    """Aggregated neighbour messages update the node state through a GRU cell.

    GGS-NN-style update (Li et al. ICLR 2016): h_i' = GRU(sum_j A[i,j] msg_j, h_i).
    The GRU's gates learn how much of the self state to keep vs how much neighbour
    input to integrate, replacing the fixed self_scale mix (self_scale is accepted
    but unused).
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
# VARIANT 5b — Convolutional update                       (model_type="update_conv")
# Idea: keep the baseline weighted-sum aggregation, but the UPDATE psi(h_i,agg_i)
# is a 1D convolution over the feature dimension instead of the additive mix. The
# self projection and the aggregated neighbour message are the two channels of a
# length-out_dim signal; a Conv1d slides a small kernel over the feature axis and
# mixes the two channels into the updated node state.
# ===========================================================================

class ConvUpdateGNNLayer(nn.Module):
    """Baseline aggregation, Conv1D-over-features update (see banner).

    self_scale is accepted for signature parity but unused: the conv learns the
    self/message mixing. At A_scale=0 the message channel is zero, so the update
    depends only on the self channel and the neighbour-ablation still holds.
    """

    def __init__(self, in_dim, out_dim, self_scale=0.5, kernel_size=3):
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
        self.conv = nn.Conv1d(in_channels=2, out_channels=1,
                              kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        batch, N, _ = x.shape
        msg = torch.matmul(A, self.W_message(x))           # (B, N, out) weighted-sum agg
        h = self.W_self(x)                                  # (B, N, out) self projection
        # two channels (self, message) along the feature axis -> conv -> one channel
        stacked = torch.stack([h, msg], dim=2).reshape(batch * N, 2, -1)  # (B*N, 2, out)
        updated = self.conv(stacked).reshape(batch, N, -1)  # (B, N, out)
        return self.relu(self.norm(updated))


class ConvUpdateGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork whose update is a Conv1D over the feature dimension."""

    def __init__(self, num_neurons, conv_kernel_size=3, **kwargs):
        self._conv_kernel_size = conv_kernel_size
        super().__init__(num_neurons, **kwargs)

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(ConvUpdateGNNLayer(in_dim, hidden_dim, self_scale=self_scale,
                                             kernel_size=self._conv_kernel_size))
        return layers

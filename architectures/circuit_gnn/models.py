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
# `encoder_type`:
#      standard             plain conv/window encoder.
#      memory_slot          + per-neuron memory concatenated (additive prior).
#      memory_film          per-neuron memory FiLM-modulates conv maps (unbounded).
#      memory_film_bounded  memory_film with tanh-bounded scale (stable).
#      memory_hybrid        bounded FiLM modulation + concatenated memory (both).
#      memory_recurrent     GRU over the window time axis, seeded by per-neuron
#                           memory: a dynamic state, not a static prior.
#      rate_norm            standardize each neuron's window by its own baseline
#                           rate so a rare source's spike is as salient as a
#                           frequent one (fixes low-rate presynaptic gradient
#                           starvation).
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
#   7  rate_aware        REGULARIZATION axis. Baseline scaffold, but the L1
#                        sparsity penalty is weighted per PRESYNAPTIC neuron by
#                        inverse firing rate, so edges out of low-rate sources
#                        are shrunk less. Targets the systematic miss of
#                        low-firing presynaptic populations (e.g. V1 in v1_v2).
#   8  msg_norm          MESSAGE-PASSING tweak. L2-normalize each source's
#                        message to unit norm before aggregation, so a source's
#                        influence is the edge weight A[i,j] alone, not its
#                        activity-scaled message norm. Parameter-free; restores
#                        gradient to low-rate sources' outgoing edges.
#
# Aggregation refs: GNN-VPA (Schneckenreiter et al. 2024) for sum/mean/max/vpa;
# GAT (Velickovic et al. 2018) for attention. Update ref: GGS-NN (Li et al. 2016)
# for the GRU cell.
#
# FILE LAYOUT (grouped by axis):
#   1. ENCODERS              — SpikeEncoder + the memory-encoder variants.
#   2. CORE SCAFFOLD         — GNNLayer + GraphNeuralNetwork (model_type=standard).
#   3. MESSAGE-PASSING TWEAKS— hadamard_mod, multi_step, msg_norm.
#   4. AGGREGATION VARIANTS  — agg_sum/mean/max/vpa + agg_attention.
#   5. UPDATE VARIANTS       — update_gru + update_conv.
#   6. INTERPRETABILITY      — InterpretableGNN + signed_input_jacobian probe.
#   7. REGULARIZATION        — RateAwareGNN (rate-weighted sparsity penalty).


# ###########################################################################
# #                                                                         #
# #                         1. ENCODERS  (encoder_type)                     #
# #                                                                         #
# # SpikeEncoder is the plain conv/window encoder (encoder_type="standard"). #
# # The variants below change ONE axis of how a persistent per-neuron memory #
# # enters the window encoding: concatenated (slot), FiLM-modulated (film /  #
# # film_bounded), both (hybrid), or recurrent state over time (recurrent).  #
# #                                                                          #
# ###########################################################################

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
# ENCODER VARIANT — FiLM per-neuron memory           (encoder_type="memory_film")
# Idea: instead of concatenating the per-neuron memory (which can only add a
# per-neuron *shift* to the embedding), let the memory MODULATE the conv feature
# maps via FiLM (Perez et al. 2018): each neuron's memory row emits a per-channel
# scale gamma_i and shift beta_i applied to its conv features before the
# activation,  feats <- (1 + gamma_i) * feats + beta_i. This adds the missing
# multiplicative path so the memory can *recalibrate* a neuron's encoding (e.g.
# normalize out its baseline rate), not just bias it. FiLM is generated by a
# linear layer initialized to zero so the encoder starts at identity (== the
# plain conv encoder) and learns modulation from there. Needs num_neurons.
# ===========================================================================

class FiLMMemorySpikeEncoder(nn.Module):
    """SpikeEncoder whose per-neuron memory FiLM-modulates the conv feature maps.

    Same conv/window path as SpikeEncoder, but a (num_neurons, mem_dim) memory
    table drives a per-channel affine modulation (gamma, beta) of the conv
    feature maps after batchnorm and before the ReLU. The FiLM generator is
    zero-initialized, so at init gamma=beta=0 and the layer reduces to the plain
    encoder; modulation is learned per neuron.
    """

    def __init__(self, window_size, num_neurons, conv_out_channels=32,
                 embedding_dim=64, kernel_size=3, mem_dim=16, bound_scale=False):
        super(FiLMMemorySpikeEncoder, self).__init__()

        self.kernel_size = kernel_size
        self.conv_out_channels = conv_out_channels
        # bound_scale=True squashes gamma through tanh so the multiplicative path
        # (1 + gamma) stays in (0, 2): no sign flips or blow-ups (encoder_type
        # "memory_film_bounded"). Default False == the original memory_film.
        self.bound_scale = bound_scale
        self.conv1d = nn.Conv1d(in_channels=1,
                                out_channels=conv_out_channels,
                                kernel_size=kernel_size,
                                padding=0)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(conv_out_channels)
        # one persistent memory row per neuron, driving the FiLM modulation
        self.neuron_memory = nn.Parameter(0.01 * torch.randn(num_neurons, mem_dim))
        # FiLM generator: memory -> per-channel (gamma, beta). Zero-init => the
        # encoder starts at identity (gamma=beta=0) and learns modulation.
        self.film = nn.Linear(mem_dim, 2 * conv_out_channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.fc = nn.Linear(conv_out_channels * window_size, embedding_dim)

    def forward(self, x):
        batch_size, num_neurons, window_size = x.shape
        x = x.view(batch_size * num_neurons, 1, window_size)
        x = F.pad(x, (self.kernel_size - 1, 0))  # left-only causal pad
        out = self.conv1d(x)
        out = self.batchnorm(out)                # (B*N, C, W)

        # per-neuron, per-channel FiLM from the memory table
        gamma_beta = self.film(self.neuron_memory)               # (N, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)                 # (N, C), (N, C)
        if self.bound_scale:
            gamma = torch.tanh(gamma)        # (1 + gamma) in (0, 2); still 0 at zero-init
        gamma = gamma.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * num_neurons, -1, 1)
        beta = beta.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * num_neurons, -1, 1)
        out = (1.0 + gamma) * out + beta         # FiLM-modulated feature maps

        out = self.relu(out)
        out = out.view(batch_size, num_neurons, -1)
        embeddings = self.fc(out)
        return embeddings


# ===========================================================================
# ENCODER VARIANT — Hybrid memory                    (encoder_type="memory_hybrid")
# Idea: combine the two winning memory encoders. The per-neuron memory table both
# (a) FiLM-modulates the conv feature maps with a BOUNDED scale (tanh gamma, so
# (1+gamma) in (0,2)) — the multiplicative recalibration of memory_film made
# stable — AND (b) is concatenated raw into the fc input — the additive per-neuron
# prior of memory_slot. So memory gets both a multiplicative and an additive path.
# The FiLM generator is zero-init, so at init the encoder == plain conv features
# (gamma=beta=0) with the raw memory concat. Needs num_neurons.
# ===========================================================================

class HybridMemorySpikeEncoder(nn.Module):
    """Per-neuron memory that BOTH FiLM-modulates the conv maps (bounded gamma)
    and is concatenated into the fc input.

    Superset of MemorySpikeEncoder (additive concat) and FiLMMemorySpikeEncoder
    (multiplicative modulation): one (num_neurons, mem_dim) table feeds a
    zero-init FiLM generator whose scale is bounded by tanh, and the same table
    row is appended to each neuron's flattened, modulated conv features.
    """

    def __init__(self, window_size, num_neurons, conv_out_channels=32,
                 embedding_dim=64, kernel_size=3, mem_dim=16):
        super(HybridMemorySpikeEncoder, self).__init__()

        self.kernel_size = kernel_size
        self.conv_out_channels = conv_out_channels
        self.conv1d = nn.Conv1d(in_channels=1,
                                out_channels=conv_out_channels,
                                kernel_size=kernel_size,
                                padding=0)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(conv_out_channels)
        # one persistent memory row per neuron, driving FiLM and concatenated raw
        self.neuron_memory = nn.Parameter(0.01 * torch.randn(num_neurons, mem_dim))
        # FiLM generator (zero-init => starts at identity modulation)
        self.film = nn.Linear(mem_dim, 2 * conv_out_channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        # +mem_dim: the raw memory row is concatenated alongside the conv features
        self.fc = nn.Linear(conv_out_channels * window_size + mem_dim, embedding_dim)

    def forward(self, x):
        batch_size, num_neurons, window_size = x.shape
        x = x.view(batch_size * num_neurons, 1, window_size)
        x = F.pad(x, (self.kernel_size - 1, 0))  # left-only causal pad
        out = self.conv1d(x)
        out = self.batchnorm(out)                # (B*N, C, W)

        # bounded per-neuron, per-channel FiLM from the memory table
        gamma_beta = self.film(self.neuron_memory)               # (N, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)                 # (N, C), (N, C)
        gamma = torch.tanh(gamma)        # (1 + gamma) in (0, 2); still 0 at zero-init
        gamma = gamma.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * num_neurons, -1, 1)
        beta = beta.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * num_neurons, -1, 1)
        out = (1.0 + gamma) * out + beta         # FiLM-modulated feature maps

        out = self.relu(out)
        out = out.view(batch_size, num_neurons, -1)
        # append each neuron's persistent memory row, broadcast over the batch
        mem = self.neuron_memory.unsqueeze(0).expand(batch_size, -1, -1)
        out = torch.cat([out, mem], dim=-1)
        embeddings = self.fc(out)
        return embeddings


# ===========================================================================
# ENCODER VARIANT — Recurrent memory               (encoder_type="memory_recurrent")
# Idea: make the per-neuron memory GENUINELY recurrent instead of a static prior.
# A GRU runs over the window's TIME axis (one step per bin), so each neuron's
# hidden state is updated bin-by-bin by its actual spike sequence — dynamic state,
# not a fixed learned vector. The persistent per-neuron memory table seeds the
# GRU's INITIAL hidden state h0[i] (the neuron's identity / baseline carried into
# the recurrence); the GRU then evolves it across the window and the FINAL hidden
# state becomes the embedding. A small causal conv first lifts each 1-D bin to a
# local feature vector that feeds the GRU. Needs num_neurons.
#
# Note: this is recurrence over the WITHIN-WINDOW time axis. Windows slide by
# step=1, so consecutive windows overlap by (window_size - 1) bins and this
# already carries almost the same history a true cross-window state would —
# without breaking the shuffled, independent-window training loop shared by the
# other variants. A true cross-batch stateful version would need a non-shuffled,
# temporally contiguous loader and detached state carried between batches.
# ===========================================================================

class RecurrentMemorySpikeEncoder(nn.Module):
    """Encoder whose per-neuron memory is a recurrent state over the time axis.

    A causal conv lifts each spike bin to a local feature vector; a GRU then runs
    over the window's bins per neuron, initialized from a persistent per-neuron
    memory table (projected to the GRU hidden size). The final hidden state is
    projected to the embedding, so the "memory" is a dynamic state updated by the
    spike sequence rather than a static learned prior.
    """

    def __init__(self, window_size, num_neurons, conv_out_channels=32,
                 embedding_dim=64, kernel_size=3, mem_dim=16,
                 gru_hidden=64, gru_layers=1):
        super(RecurrentMemorySpikeEncoder, self).__init__()

        self.kernel_size = kernel_size
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.num_neurons = num_neurons

        # per-time-step local features feeding the GRU (causal pad keeps length=W)
        self.conv1d = nn.Conv1d(in_channels=1,
                                out_channels=conv_out_channels,
                                kernel_size=kernel_size,
                                padding=0)
        self.batchnorm = nn.BatchNorm1d(conv_out_channels)
        self.relu = nn.ReLU()
        # persistent per-neuron memory -> GRU initial hidden state (the identity)
        self.neuron_memory = nn.Parameter(0.01 * torch.randn(num_neurons, mem_dim))
        self.h0_proj = nn.Linear(mem_dim, gru_layers * gru_hidden)
        self.gru = nn.GRU(input_size=conv_out_channels, hidden_size=gru_hidden,
                          num_layers=gru_layers, batch_first=True)
        self.fc = nn.Linear(gru_hidden, embedding_dim)

    def forward(self, x):
        batch_size, num_neurons, window_size = x.shape
        x = x.view(batch_size * num_neurons, 1, window_size)
        x = F.pad(x, (self.kernel_size - 1, 0))  # left-only causal pad
        out = self.conv1d(x)                      # (B*N, C, W)
        out = self.batchnorm(out)
        out = self.relu(out)
        seq = out.transpose(1, 2).contiguous()    # (B*N, W, C): sequence over time

        # per-neuron memory -> initial hidden state, broadcast over the batch.
        # Row order of seq is batch-major (b*N + n), so h0 must match: b outer,
        # n inner.
        h0 = self.h0_proj(self.neuron_memory)                          # (N, L*H)
        h0 = h0.view(num_neurons, self.gru_layers, self.gru_hidden)    # (N, L, H)
        h0 = h0.permute(1, 0, 2).contiguous()                         # (L, N, H)
        h0 = h0.unsqueeze(1).expand(-1, batch_size, -1, -1)          # (L, B, N, H)
        h0 = h0.reshape(self.gru_layers, batch_size * num_neurons,
                        self.gru_hidden).contiguous()                 # (L, B*N, H)

        _, hn = self.gru(seq, h0)                  # hn: (L, B*N, H)
        last = hn[-1]                              # (B*N, H): final state, top layer
        embeddings = self.fc(last)                 # (B*N, embedding_dim)
        embeddings = embeddings.view(batch_size, num_neurons, -1)
        return embeddings


# ===========================================================================
# ENCODER VARIANT — Rate normalization               (encoder_type="rate_norm")
# Idea: the baseline conv path (and its cross-neuron BatchNorm) represents a
# low-firing neuron's spikes with a tiny activation relative to high-firing
# neurons, so a rare presynaptic spike is washed out and its outgoing edges are
# gradient-starved — no signal reaches A[i,j] no matter how the penalty is tuned.
# Fix: standardize each neuron's window by its OWN baseline BEFORE the conv:
#      x <- (x - mu_i) / sqrt(mu_i + eps),   mu_i = neuron i's mean spikes/bin.
# A spike from a low-rate neuron then becomes a large value (~1/sqrt(mu_i)) rather
# than a small one, so its conv features stand out and gradient flows onto its
# outgoing edges. sqrt(mu_i) is the Poisson scale; eps floors the amplification for
# near-silent neurons. Because all neurons now share a common input scale, the
# downstream cross-neuron BatchNorm is no longer dominated by high-rate neurons, so
# the rest of the path is identical to SpikeEncoder. Needs num_neurons + firing_rates.
#
# This is the "inject signal" complement to the rate_aware penalty ("remove the
# shrinkage headwind"): rate_aware only lets an edge survive if it already has some
# data gradient, whereas rate_norm creates that gradient. The two are composable.
# ===========================================================================

class RateNormSpikeEncoder(nn.Module):
    """SpikeEncoder that standardizes each neuron's window by its own baseline rate.

    Identical conv/window path to SpikeEncoder, but the input is first standardized
    per neuron, x <- (x - mu_i) / sqrt(mu_i + eps) with mu_i the neuron's mean
    spikes/bin, so a rare source's spike produces a large activation instead of
    being washed out. firing_rates=None falls back to identity (== SpikeEncoder).
    """

    def __init__(self, window_size, num_neurons, firing_rates=None,
                 conv_out_channels=32, embedding_dim=64, kernel_size=3, eps=1e-3):
        super(RateNormSpikeEncoder, self).__init__()

        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(in_channels=1,
                                out_channels=conv_out_channels,
                                kernel_size=kernel_size,
                                padding=0)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(conv_out_channels)
        self.fc = nn.Linear(conv_out_channels * window_size, embedding_dim)

        # per-neuron standardization constants. Buffers: move with .to(device) and
        # are saved in the checkpoint. None -> identity (behaves like SpikeEncoder).
        if firing_rates is None:
            mu = torch.zeros(num_neurons)
            denom = torch.ones(num_neurons)
        else:
            mu = torch.as_tensor(firing_rates, dtype=torch.float32).flatten()
            assert mu.numel() == num_neurons, \
                f"firing_rates has {mu.numel()} entries, expected {num_neurons}"
            denom = torch.sqrt(mu + eps)          # Poisson scale, eps floors the gain
        self.register_buffer("rate_mean", mu)
        self.register_buffer("rate_denom", denom)

    def forward(self, x):
        batch_size, num_neurons, window_size = x.shape
        # standardize each neuron by its own baseline rate (broadcast over batch/time)
        x = (x - self.rate_mean.view(1, -1, 1)) / self.rate_denom.view(1, -1, 1)
        x = x.reshape(batch_size * num_neurons, 1, window_size)
        x = F.pad(x, (self.kernel_size - 1, 0))  # left-only causal pad
        out = self.conv1d(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = out.view(batch_size * num_neurons, -1)
        embeddings = self.fc(out)
        embeddings = embeddings.view(batch_size, num_neurons, -1)
        return embeddings


# ###########################################################################
# #                                                                         #
# #               2. CORE SCAFFOLD  (model_type="standard")                 #
# #                                                                         #
# # The baseline message-passing GNN. Encode each neuron's spike window,    #
# # learn a signed N x N adjacency, and message-pass `num_layers` times to  #
# # predict the next bin. Every model variant below is a single-axis tweak  #
# # of this scaffold, overriding only _build_layers unless noted.           #
# #                                                                         #
# ###########################################################################

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


# ###########################################################################
# #                                                                         #
# #         3. MESSAGE-PASSING TWEAKS  (model_type=hadamard_mod, multi_step) #
# #                                                                         #
# # Single-axis variants of the core scaffold that change the message       #
# # function or the prediction horizon (not the aggregation/update axes).   #
# #                                                                         #
# ###########################################################################

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
# VARIANT 8 — Message normalization                     (model_type="msg_norm")
# Idea: in the baseline layer the per-edge contribution is A[i,j] * m_j, where
# m_j = W_message(x_j). The message norm ||m_j|| scales with how active neuron j
# is (an active source produces a larger embedding -> larger message), so the
# gradient onto A[i,j] is itself proportional to ||m_j||: a low-rate source gets a
# tiny gradient and its outgoing edge stays faint or dies, no matter how the input
# or the penalty is scaled. This is the activity confound that rate normalization
# of the INPUT cannot remove, because W_message re-introduces it at every layer.
#
# Fix: L2-normalize each source's message to unit norm before aggregation,
#      neighbor_agg[i] = sum_j A[i,j] * (m_j / ||m_j||).
# Now every source contributes at the same scale, so the influence of j on i — and
# the gradient onto A[i,j] — is carried by the edge weight ALONE, decoupled from
# j's activity. This is exactly what structure inference wants: detect that j->i
# exists regardless of how loudly j fires. Parameter-free (the eps is a fixed
# numerical floor against a zero-norm message, not a dataset knob), so nothing to
# tune per dataset. Only the neighbour messages are normalized; the self term is
# untouched, and A_scale=0 still zeros the neighbour path (ablation intact).
# ===========================================================================

class MsgNormGNNLayer(nn.Module):
    """Baseline layer that L2-normalizes each source's message before aggregation.

    Identical to GNNLayer except each neuron's message m_j is divided by its own
    L2 norm, so sum_j A[i,j] m_j becomes sum_j A[i,j] (m_j / ||m_j||): the edge
    weight A[i,j] carries the influence, decoupled from source activity. eps is a
    fixed numerical floor (not a tunable hyperparameter).
    """

    def __init__(self, in_dim, out_dim, self_scale=0.5, eps=1e-8):
        super().__init__()
        mlp_dim = 2 * in_dim
        self.self_scale = self_scale
        self.eps = eps
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
        messages = self.W_message(x)                              # (B, N, out_dim)
        # unit-norm each source neuron's message (per batch, per neuron)
        messages = messages / (messages.norm(dim=-1, keepdim=True) + self.eps)
        self_term = self.W_self(x)
        neighbor_agg = torch.matmul(A, messages)                  # sum_j A[i,j] m̂_j
        out = self.relu(self.norm(neighbor_agg + self.self_scale * self_term))
        return out


class MsgNormGNN(GraphNeuralNetwork):
    """GraphNeuralNetwork whose every layer unit-norms messages before aggregation."""

    def _build_layers(self, num_layers, embedding_dim, hidden_dim, self_scale):
        layers = []
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            layers.append(MsgNormGNNLayer(in_dim, hidden_dim, self_scale=self_scale))
        return layers


# ###########################################################################
# #                                                                         #
# #   4. AGGREGATION VARIANTS                                               #
# #      (model_type="agg_sum"|"agg_mean"|"agg_max"|"agg_vpa"|"agg_attention")#
# #                                                                         #
# # All change ONE axis: how neighbour messages are reduced over the        #
# # sources j. sum/mean/max/vpa keep the learned signed A and only change   #
# # the reduction (GNN-VPA, Schneckenreiter et al. 2024); attention REPLACES #
# # A with GAT weights (Velickovic et al. 2018) and reads connectivity out  #
# # of the running mean attention.                                          #
# #                                                                         #
# ###########################################################################

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


# ###########################################################################
# #                                                                         #
# #     5. UPDATE VARIANTS  (model_type="update_gru" | "update_conv")       #
# #                                                                         #
# # All change ONE axis: how the aggregated neighbour message integrates    #
# # into the node state. update_gru uses a GRU cell (GGS-NN, Li et al.      #
# # 2016); update_conv uses a Conv1D over the feature dim. Both keep the    #
# # baseline weighted-sum aggregation and the learned signed A.             #
# #                                                                         #
# ###########################################################################

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


# ###########################################################################
# #                                                                         #
# #         6. INTERPRETABILITY  (model_type="interpretable")               #
# #                                                                         #
# # Behaviourally identical to the baseline (same scaffold, same training)  #
# # but adds analysis probes that read effective connectivity OUT of the    #
# # trained message passing (Jacobian connectivity & mono-vs-polysynaptic   #
# # split), plus a module-level signed-Jacobian probe through the full      #
# # encoder+GNN pipeline.                                                   #
# #                                                                         #
# ###########################################################################

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


# ###########################################################################
# #                                                                         #
# #      7. REGULARIZATION  (model_type="rate_aware")                       #
# #                                                                         #
# # Same scaffold, message passing, and signed adjacency as the baseline;   #
# # the ONLY change is the L1 sparsity penalty, which is weighted per        #
# # presynaptic neuron by inverse firing rate (see banner below). Fixes the #
# # systematic miss of low-firing presynaptic populations.                  #
# #                                                                         #
# ###########################################################################

# ===========================================================================
# VARIANT 7 — Rate-aware sparsity penalty              (model_type="rate_aware")
# Idea: the edge A[i,j] (j = presynaptic source, i = postsynaptic target) only
# receives an upward gradient in windows where source j actually spikes, so that
# gradient scales with j's firing rate. Meanwhile the L1 sparsity penalty applies
# a constant, rate-blind shrinkage to every edge on every step. Low-rate sources
# therefore lose the tug-of-war and their outgoing edges die at zero — exactly the
# "V2 presynaptic edges found, V1 presynaptic edges missed" failure.
#
# Fix: weight the L1 shrinkage of each PRESYNAPTIC column j by w_j ∝ 1/(rate_j+eps),
# normalized to mean 1 (so the average shrinkage — and thus the effective global
# sparsity budget — matches the baseline) and clamped so a near-silent neuron does
# not escape sparsity entirely. Edges out of low-rate sources are shrunk less and
# can survive; high-rate sources see ~baseline pressure. The competitive penalty is
# left rate-uniform (it only arbitrates the exc/inh sign, which is rate-independent).
#
# The per-column weight is a buffer read by train.compute_loss; a model without it
# falls back to the uniform baseline penalty, so this class is the only opt-in.
# Needs the per-neuron firing rates at construction (train_model passes them).
# ===========================================================================

class RateAwareGNN(GraphNeuralNetwork):
    """Baseline GNN with a presynaptic-rate-weighted L1 sparsity penalty.

    Identical to GraphNeuralNetwork in every forward/aggregation respect; it only
    carries ``penalty_col_weight`` — a length-N buffer of per-presynaptic-neuron
    (column) L1 weights ∝ 1/(rate+eps), mean-normalized and clamped — which
    train.compute_loss multiplies into the per-column L1 shrinkage.

    firing_rates : 1-D array/tensor of length num_neurons, each neuron's mean
        spikes-per-bin (or any monotone activity measure). None → uniform weights
        (== baseline). rate_eps : additive floor; None → mean of the rates, which
        keeps the weighting scale-adaptive. weight_clip : cap on any single column
        weight after normalization so a near-silent source keeps some sparsity.
    """

    def __init__(self, num_neurons, firing_rates=None, rate_eps=None,
                 weight_clip=5.0, **kwargs):
        super().__init__(num_neurons, **kwargs)
        w = self._compute_col_weights(num_neurons, firing_rates, rate_eps, weight_clip)
        # per-presynaptic-neuron (column) L1 weight; read by train.compute_loss
        self.register_buffer("penalty_col_weight", w)

    @staticmethod
    def _compute_col_weights(num_neurons, firing_rates, rate_eps, weight_clip):
        if firing_rates is None:
            return torch.ones(num_neurons)
        r = torch.as_tensor(firing_rates, dtype=torch.float32).flatten()
        assert r.numel() == num_neurons, \
            f"firing_rates has {r.numel()} entries, expected {num_neurons}"
        # scale-adaptive floor: default eps = mean rate (>=0). Guard the all-silent
        # edge case so we never divide by zero.
        eps = float(rate_eps) if rate_eps is not None else float(r.mean())
        if eps <= 0:
            eps = 1.0
        w = 1.0 / (r + eps)
        w = w / w.mean()                       # mean 1 → same global sparsity budget
        if weight_clip is not None:
            w = w.clamp(max=float(weight_clip))
            w = w / w.mean()                   # renormalize after clamping
        return w

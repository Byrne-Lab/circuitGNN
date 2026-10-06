import subprocess
import sys

import torch

from circuit_gnn import GraphNeuralNetwork, SpikeEncoder


def test_import_does_not_pull_plotting_stack():
    """`import circuit_gnn` must not import matplotlib/seaborn (see the package docstring)."""
    code = ("import circuit_gnn, sys; "
            "print([m for m in ('matplotlib', 'seaborn') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_forward_shapes():
    n, window, batch = 5, 20, 3
    encoder = SpikeEncoder(window_size=window, num_neurons=n, embedding_dim=8,
                           conv_out_channels=4, mem_dim=2)
    gnn = GraphNeuralNetwork(num_neurons=n, embedding_dim=8, hidden_dim=8, num_layers=2)

    embeddings = encoder(torch.rand(batch, n, window))
    assert embeddings.shape == (batch, n, 8)
    assert gnn(embeddings).shape == (batch, n)


def test_signed_adjacency_has_zero_diagonal():
    gnn = GraphNeuralNetwork(num_neurons=4, embedding_dim=8, hidden_dim=8, num_layers=1)
    A = gnn.signed_adjacency()
    assert torch.allclose(torch.diagonal(A), torch.zeros(4))
    expected = torch.relu(gnn.learned_adj_exc) - torch.relu(gnn.learned_adj_inh)
    off = ~torch.eye(4, dtype=torch.bool)
    assert torch.equal(A[off], expected[off])


def test_zeroed_adjacency_isolates_neurons():
    """The ablation test zeroes A: each neuron's output must then ignore the others."""
    gnn = GraphNeuralNetwork(num_neurons=3, embedding_dim=8, hidden_dim=8, num_layers=2)
    x = torch.rand(1, 3, 8)
    x_other = x.clone()
    x_other[:, 1:] += 1.0                       # perturb every neuron except neuron 0

    with torch.no_grad():
        assert not torch.allclose(gnn(x)[0, 0], gnn(x_other)[0, 0])
        gnn.A_scale = 0.0
        assert torch.allclose(gnn(x)[0, 0], gnn(x_other)[0, 0])

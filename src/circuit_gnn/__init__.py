"""Inference of signed synaptic connectivity from spike trains with a graph neural network.

Plotting helpers live in circuit_gnn.plotting; they are deliberately not re-exported
here so that importing this package stays light.
"""
from .binning import optimal_bin_ms, rebin, spikes_from_times
from .consensus import (Consensus, consensus, get_consistent_connections, load_runs,
                        orient_runs, robust_sign_inference)
from .infer import InferenceResult, fit_runs, infer_connectivity
from .metrics import calculate_fit
from .models import GNNLayer, GraphNeuralNetwork, SpikeEncoder
from .train import Config, SpikeTrainDataset, train_model

__version__ = "0.1.0"

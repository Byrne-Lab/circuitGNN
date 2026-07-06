"""Variant sweep runner for circuitGNN.

Run from repo root:
    python variants/run_variants.py --datasets X --variants Y --num_runs Z

Results are written to variants/results/{dataset}/{variant}/.
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

import torch
import numpy as np

VARIANTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = VARIANTS_DIR.parent
sys.path.insert(0, str(VARIANTS_DIR))
os.chdir(str(REPO_ROOT))

from circuit_gnn.data import load_yuto_network, load_pdcm_network, load_pdcm_large_network, load_arbitrary_network, load_small_circuit_A_network, load_small_circuit_B_network, load_mdp_network, load_v1_v2_network, load_hc3_session
from circuit_gnn.train import TrainConfig, train_model, load_checkpoint
from circuit_gnn.utils import plot_connectivity_matrix, plot_pdcm_matrix, plot_mdp_matrix, plot_v1_v2_matrix, plot_hc3_matrix

RESULTS_BASE = VARIANTS_DIR / "results"

# variant definitions
# config: TrainConfig field overrides
# encoder_kwargs / gnn_kwargs: forwarded to build_models

_BASE_ENC = {"conv_out_channels": 32, "embedding_dim": 64, "kernel_size": 3}
_BASE_GNN = {"embedding_dim": 64, "hidden_dim": 64, "output_dim": 1, "num_layers": 5, "desired_max": 0.05, "init_noise": 0.01}
_BASE_CFG = {"batch_size": 16, "l1_lambda": 7.5, "competitive_lambda": 15.0}

# Each variant may contain:
#   config              — TrainConfig field overrides (batch_size, window_size, l1_lambda, ...)
#   encoder_kwargs      — passed to SpikeEncoder (conv_out_channels, kernel_size, embedding_dim)
#   gnn_kwargs          — passed to GraphNeuralNetwork (num_layers, hidden_dim, ...)
#   downsample_override — {dataset: factor} map; dataset is skipped if not present in map

VARIANTS = {

    # batch size
    "batch_4":        {"config": {**_BASE_CFG, "batch_size": 4},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "batch_16":       {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "batch_32":       {"config": {**_BASE_CFG, "batch_size": 32},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "batch_64":       {"config": {**_BASE_CFG, "batch_size": 64},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},

    # number of GNN layers
    "layers_1":       {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": {**_BASE_GNN, "num_layers": 1}},
    "layers_3":       {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": {**_BASE_GNN, "num_layers": 3}},
    "layers_5":       {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "layers_7":       {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": {**_BASE_GNN, "num_layers": 7}},

    # bin size via downsample factor
    "bin_1ms":        {"config": {**_BASE_CFG, "num_epochs": 50},
                       "downsample_override": {"yuto": 2, "pdcm": 1, "pdcm_large": 1, "mdp": 1, "v1_v2": 1, "arbitrary": 1, "small_circuit_A": 2, "small_circuit_B": 2},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "bin_2ms":        {"config": {**_BASE_CFG, "num_epochs": 50},
                       "downsample_override": {"yuto": 4, "pdcm": 2, "pdcm_large": 2, "mdp": 2, "v1_v2": 2, "arbitrary": 2, "small_circuit_A": 4, "small_circuit_B": 4},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "bin_3ms":        {"config": {**_BASE_CFG, "num_epochs": 50},
                       "downsample_override": {"yuto": 6, "pdcm": 3, "pdcm_large": 3, "mdp": 3, "v1_v2": 3, "arbitrary": 3, "small_circuit_A": 6, "small_circuit_B": 6},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "bin_4ms":        {"config": {**_BASE_CFG, "num_epochs": 50},
                       "downsample_override": {"yuto": 8, "pdcm": 4, "pdcm_large": 4, "mdp": 4, "v1_v2": 4, "arbitrary": 4, "small_circuit_A": 8, "small_circuit_B": 8},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "bin_5ms":        {"config": {**_BASE_CFG, "num_epochs": 50},
                       "downsample_override": {"yuto": 10, "pdcm": 5, "pdcm_large": 5, "mdp": 5, "v1_v2": 5, "arbitrary": 5, "small_circuit_A": 10, "small_circuit_B": 10},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "bin_6ms":        {"config": {**_BASE_CFG, "num_epochs": 50},
                       "downsample_override": {"yuto": 12, "pdcm": 6, "pdcm_large": 6, "mdp": 6, "v1_v2": 6, "arbitrary": 6, "small_circuit_A": 12, "small_circuit_B": 12},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "bin_7ms":        {"config": {**_BASE_CFG, "num_epochs": 50},
                       "downsample_override": {"yuto": 14, "pdcm": 7, "pdcm_large": 7, "mdp": 7, "v1_v2": 7, "arbitrary": 7, "small_circuit_A": 14, "small_circuit_B": 14},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "bin_8ms":        {"config": {**_BASE_CFG, "num_epochs": 50},
                       "downsample_override": {"yuto": 16, "pdcm": 8, "pdcm_large": 8, "mdp": 8, "v1_v2": 8, "arbitrary": 8, "small_circuit_A": 16, "small_circuit_B": 16},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "bin_9ms":        {"config": {**_BASE_CFG, "num_epochs": 50},
                       "downsample_override": {"yuto": 18, "pdcm": 9, "pdcm_large": 9, "mdp": 9, "v1_v2": 9, "arbitrary": 9, "small_circuit_A": 18, "small_circuit_B": 18},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "bin_10ms":       {"config": {**_BASE_CFG, "num_epochs": 50},
                       "downsample_override": {"yuto": 20, "pdcm": 10, "pdcm_large": 10, "mdp": 10, "v1_v2": 10, "arbitrary": 10, "small_circuit_A": 20, "small_circuit_B": 20},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "bin_20ms":       {"config": {**_BASE_CFG, "num_epochs": 50},
                       "downsample_override": {"yuto": 40, "pdcm": 20, "pdcm_large": 20, "mdp": 20, "v1_v2": 20, "arbitrary": 20, "small_circuit_A": 40, "small_circuit_B": 40},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},

    # window size
    "window_5":       {"config": {**_BASE_CFG, "window_size": 5},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "window_10":      {"config": {**_BASE_CFG, "window_size": 10},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "window_50":      {"config": {**_BASE_CFG, "window_size": 50},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "window_100":     {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "window_200":     {"config": {**_BASE_CFG, "window_size": 200},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},

    # encoder conv channels
    "channels_8":     {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": {**_BASE_ENC, "conv_out_channels": 8},
                       "gnn_kwargs": dict(**_BASE_GNN)},
    "channels_16":    {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": {**_BASE_ENC, "conv_out_channels": 16},
                       "gnn_kwargs": dict(**_BASE_GNN)},
    "channels_32":    {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "channels_64":    {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": {**_BASE_ENC, "conv_out_channels": 64},
                       "gnn_kwargs": dict(**_BASE_GNN)},

    # encoder kernel size
    "kernel_size_2":  {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": {**_BASE_ENC, "kernel_size": 2},
                       "gnn_kwargs": dict(**_BASE_GNN)},
    "kernel_size_3":  {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "kernel_size_4":  {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": {**_BASE_ENC, "kernel_size": 4},
                       "gnn_kwargs": dict(**_BASE_GNN)},

    # L1 regularisation on adjacency (sparsity)
    "l1_0p5":         {"config": {**_BASE_CFG, "l1_lambda": 0.5},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_1":           {"config": {**_BASE_CFG, "l1_lambda": 1.0},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_1p5":         {"config": {**_BASE_CFG, "l1_lambda": 1.5},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_2":           {"config": {**_BASE_CFG, "l1_lambda": 2.0},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_2p5":         {"config": {**_BASE_CFG, "l1_lambda": 2.5},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_5":           {"config": {**_BASE_CFG, "l1_lambda": 5.0},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_7p5":         {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_10":          {"config": {**_BASE_CFG, "l1_lambda": 10.0},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_15":          {"config": {**_BASE_CFG, "l1_lambda": 15.0},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},

    # competitive lambda (exc/inh separation)
    "comp_5":         {"config": {**_BASE_CFG, "competitive_lambda": 5.0},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "comp_10":        {"config": {**_BASE_CFG, "competitive_lambda": 10.0},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "comp_15":        {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "comp_20":        {"config": {**_BASE_CFG, "competitive_lambda": 20.0},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "comp_30":        {"config": {**_BASE_CFG, "competitive_lambda": 30.0},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},

    # L1 × competitive lambda sweep
    "l1_1_comp_5":      {"config": {**_BASE_CFG, "l1_lambda": 1.0, "competitive_lambda": 5.0},
                         "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_1_comp_10":     {"config": {**_BASE_CFG, "l1_lambda": 1.0, "competitive_lambda": 10.0},
                         "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_1_comp_15":     {"config": {**_BASE_CFG, "l1_lambda": 1.0, "competitive_lambda": 15.0},
                         "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_1p5_comp_5":    {"config": {**_BASE_CFG, "l1_lambda": 1.5, "competitive_lambda": 5.0},
                         "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_1p5_comp_10":   {"config": {**_BASE_CFG, "l1_lambda": 1.5, "competitive_lambda": 10.0},
                         "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_1p5_comp_15":   {"config": {**_BASE_CFG, "l1_lambda": 1.5, "competitive_lambda": 15.0},
                         "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_2_comp_5":      {"config": {**_BASE_CFG, "l1_lambda": 2.0, "competitive_lambda": 5.0},
                         "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_2_comp_10":     {"config": {**_BASE_CFG, "l1_lambda": 2.0, "competitive_lambda": 10.0},
                         "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_2_comp_15":     {"config": {**_BASE_CFG, "l1_lambda": 2.0, "competitive_lambda": 15.0},
                         "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_2p5_comp_5":    {"config": {**_BASE_CFG, "l1_lambda": 2.5, "competitive_lambda": 5.0},
                         "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_2p5_comp_10":   {"config": {**_BASE_CFG, "l1_lambda": 2.5, "competitive_lambda": 10.0},
                         "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "l1_2p5_comp_15":   {"config": {**_BASE_CFG, "l1_lambda": 2.5, "competitive_lambda": 15.0},
                         "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},

    # self_scale in GNNLayer
    "self_0p1":       {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": {**_BASE_GNN, "self_scale": 0.1}},
    "self_0p25":      {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": {**_BASE_GNN, "self_scale": 0.25}},
    "self_0p5":       {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "self_1":         {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": {**_BASE_GNN, "self_scale": 1.0}},

    # embedding dim (encoder → GNN dimension) 
    "emb_16":         {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 16},
                       "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 16}},
    "emb_32":         {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 32},
                       "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 32}},
    "emb_64":         {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "emb_128":        {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 128},
                       "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 128}},

    # GNN hidden dim
    "hidden_32":      {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": {**_BASE_GNN, "hidden_dim": 32}},
    "hidden_64":      {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "hidden_128":     {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": {**_BASE_GNN, "hidden_dim": 128}},

    # learning rate
    "lr_1e-5":        {"config": {**_BASE_CFG, "lr": 1e-5},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "lr_5e-5":        {"config": {**_BASE_CFG, "lr": 5e-5},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "lr_1e-4":        {"config": dict(**_BASE_CFG),
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    "lr_5e-4":        {"config": {**_BASE_CFG, "lr": 5e-4},
                       "encoder_kwargs": dict(**_BASE_ENC), "gnn_kwargs": dict(**_BASE_GNN)},
    
    # COMBINATION: encoder conv channels and kernel size
    "kernel_2_channels_8": {"config": dict(**_BASE_CFG),
                            "encoder_kwargs": {**_BASE_ENC, "kernel_size": 2, "conv_out_channels": 8}, "gnn_kwargs": dict(**_BASE_GNN)},
    "kernel_2_channels_16": {"config": dict(**_BASE_CFG),
                            "encoder_kwargs": {**_BASE_ENC, "kernel_size": 2, "conv_out_channels": 16}, "gnn_kwargs": dict(**_BASE_GNN)}, 
    "kernel_2_channels_32": {"config": dict(**_BASE_CFG),
                            "encoder_kwargs": {**_BASE_ENC, "kernel_size": 2, "conv_out_channels": 32}, "gnn_kwargs": dict(**_BASE_GNN)}, 
    "kernel_2_channels_64": {"config": dict(**_BASE_CFG),
                            "encoder_kwargs": {**_BASE_ENC, "kernel_size": 2, "conv_out_channels": 64}, "gnn_kwargs": dict(**_BASE_GNN)}, 

    "kernel_3_channels_8": {"config": dict(**_BASE_CFG),
                            "encoder_kwargs": {**_BASE_ENC, "kernel_size": 3, "conv_out_channels": 8}, "gnn_kwargs": dict(**_BASE_GNN)},
    "kernel_3_channels_16": {"config": dict(**_BASE_CFG),
                            "encoder_kwargs": {**_BASE_ENC, "kernel_size": 3, "conv_out_channels": 16}, "gnn_kwargs": dict(**_BASE_GNN)}, 
    "kernel_3_channels_32": {"config": dict(**_BASE_CFG),
                            "encoder_kwargs": {**_BASE_ENC, "kernel_size": 3, "conv_out_channels": 32}, "gnn_kwargs": dict(**_BASE_GNN)}, 
    "kernel_3_channels_64": {"config": dict(**_BASE_CFG),
                             "encoder_kwargs": {**_BASE_ENC, "kernel_size": 3, "conv_out_channels": 64}, "gnn_kwargs": dict(**_BASE_GNN)},

    # kernel size 4
    "kernel_4_channels_8": {"config": dict(**_BASE_CFG),
                            "encoder_kwargs": {**_BASE_ENC, "kernel_size": 4, "conv_out_channels": 8}, "gnn_kwargs": dict(**_BASE_GNN)},
    "kernel_4_channels_16": {"config": dict(**_BASE_CFG),
                            "encoder_kwargs": {**_BASE_ENC, "kernel_size": 4, "conv_out_channels": 16},  "gnn_kwargs": dict(**_BASE_GNN)}, 
    "kernel_4_channels_32": {"config": dict(**_BASE_CFG),
                            "encoder_kwargs": {**_BASE_ENC, "kernel_size": 4, "conv_out_channels": 32},  "gnn_kwargs": dict(**_BASE_GNN)}, 
    "kernel_4_channels_64": {"config": dict(**_BASE_CFG),
                            "encoder_kwargs": {**_BASE_ENC, "kernel_size": 4, "conv_out_channels": 64}, "gnn_kwargs": dict(**_BASE_GNN)},
    
    # COMBINATION: embedding dim and hidden dim
    "emb_16_hidden_16": {"config": dict(**_BASE_CFG),
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 16}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 16, "hidden_dim": 16}},
    "emb_16_hidden_32": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 16}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 16, "hidden_dim": 32}}, 
    "emb_16_hidden_64": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 16}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 16, "hidden_dim": 64}},
    "emb_16_hidden_128": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 16}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 16, "hidden_dim": 128}},

    "emb_32_hidden_16": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 32}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 32, "hidden_dim": 16}}, 
    "emb_32_hidden_32": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 32}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 32, "hidden_dim": 32}}, 
    "emb_32_hidden_64": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 32}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 32, "hidden_dim": 64}},
    "emb_32_hidden_128": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 32}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 32, "hidden_dim": 128}},

    "emb_64_hidden_16": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 64}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 64, "hidden_dim": 16}}, 
    "emb_64_hidden_32": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 64}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 64, "hidden_dim": 32}}, 
    "emb_64_hidden_64": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 64}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 64, "hidden_dim": 64}}, 
    "emb_64_hidden_128": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 64}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 64, "hidden_dim": 128}}, 

    "emb_128_hidden_16": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 128}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 128, "hidden_dim": 16}}, 
    "emb_128_hidden_32": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 128}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 128, "hidden_dim": 32}},
    "emb_128_hidden_64": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 128}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 128, "hidden_dim": 64}},
    "emb_128_hidden_128": {"config": dict(**_BASE_CFG), 
                        "encoder_kwargs": {**_BASE_ENC, "embedding_dim": 128}, "gnn_kwargs": {**_BASE_GNN, "embedding_dim": 128, "hidden_dim": 128}},
}

# dataset registry
DATASET_REGISTRY = {
    "yuto": {
        "loader": load_yuto_network,
        "loader_kwargs": {"downsample_factor": 20, "gap": 0.20, "step": 1},
        "num_epochs": 20,
        "default_window": 100,
    },
    "arbitrary": {
        "loader": load_arbitrary_network,
        "loader_kwargs": {"downsample_factor": 10},
        "num_epochs": 20,
        "default_window": 100,
    },
    "pdcm": {
        "loader": load_pdcm_network,
        "loader_kwargs": {"downsample_factor": 2},
        "num_epochs": 50,
        "default_window": 100,
    },
    "pdcm_large": {
        "loader": load_pdcm_large_network,
        "loader_kwargs": {"downsample_factor": 6},
        "num_epochs": 50,
        "default_window": 100,
    },
    "mdp": {
        "loader": load_mdp_network,
        "loader_kwargs": {"downsample_factor": 6},
        "num_epochs": 50,
        "default_window": 100,
    },
    "v1_v2": {
        "loader": load_v1_v2_network,
        "loader_kwargs": {"downsample_factor": 6},
        "num_epochs": 50,
        "default_window": 100,
    },
    "small_circuit_A": {
        "loader": load_small_circuit_A_network,
        "loader_kwargs": {"downsample_factor": 20},
        "num_epochs": 20,
        "default_window": 100,
    },
    "small_circuit_B": {
        "loader": load_small_circuit_B_network,
        "loader_kwargs": {"downsample_factor": 20},
        "num_epochs": 20,
        "default_window": 100,
    },
    # in-vivo hc-3 session (no ground truth); same session/window as the baselines.
    # downsample_factor='auto' is resolved inside the loader (native res 1 ms).
    "hc3": {
        "loader": load_hc3_session,
        "loader_kwargs": {"downsample_factor": "auto", "tar_path": "datasets/hc3/ec016.1016.tar.gz", "topdir": "ec016.58", "cell_csv": "datasets/hc3/hc3-cell.csv", "regions": ["CA3", "DG", "EC2", "EC3"], "quality_filter": True, "max_time_s": 2000},
        "num_epochs": 20,
        "default_window": 100,
    },
}


# helpers

def make_criterion(spike_tensor: torch.Tensor, device) -> torch.nn.Module:
    pos = spike_tensor.sum().item()
    neg = spike_tensor.numel() - pos
    pos_weight = torch.tensor(neg / pos, device=device) if pos > 0 else torch.tensor(1.0, device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def extract_adjacency(gnn: torch.nn.Module) -> np.ndarray:
    with torch.no_grad():
        return gnn.get_adjacency().cpu().numpy().T


def eval_with_scale(encoder, gnn, val_loader, criterion, device, A_scale: float) -> float:
    original = gnn.A_scale
    gnn.A_scale = A_scale
    total = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = gnn(encoder(inputs))
            total += criterion(outputs, targets).item() * inputs.size(0)
    gnn.A_scale = original
    return total / len(val_loader.dataset)


# main run logic

def run_dataset_variant(
    dataset_name: str,
    variant_name: str,
    dataset_cfg: dict,
    variant_spec: dict,
    num_runs: int,
    device,
) -> None:
    output_dir = RESULTS_BASE / dataset_name / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve effective downsample_factor
    if "downsample_override" in variant_spec:
        override = variant_spec["downsample_override"]
        if dataset_name not in override:
            print(f"  Skipping {dataset_name} (not in downsample_override for this bin-size variant)")
            return
        effective_ds = override[dataset_name]
    else:
        base_ds = dataset_cfg["loader_kwargs"]["downsample_factor"]
        if isinstance(base_ds, str):
            effective_ds = base_ds  # e.g. 'auto' — resolved by occupancy inside the loader
        else:
            effective_ds = max(1, round(base_ds * variant_spec.get("downsample_scale", 1.0)))
    loader_kwargs = {**dataset_cfg["loader_kwargs"], "downsample_factor": effective_ds}

    print(f"  Loading {dataset_name} data (downsample_factor={effective_ds})...")
    binned_spikes, true_conn, neuron_names = dataset_cfg["loader"](**loader_kwargs)
    num_channels, num_time = binned_spikes.shape
    spike_tensor = torch.tensor(binned_spikes, dtype=torch.float32)
    criterion = make_criterion(spike_tensor, device)

    window_size = variant_spec["config"].get("window_size", dataset_cfg["default_window"])
    pop_labels = np.load(f'datasets/{dataset_name}/pop_labels.npy') if dataset_name in ('pdcm', 'pdcm_large') else None

    # Initialise ablation CSV if needed
    csv_path = output_dir / "ablation.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["run_id", "A_val_loss", "no_A_val_loss", "difference"])

    for run_idx in range(1, num_runs + 1):
        print(f"\n  Run {run_idx}/{num_runs}")
        t0 = time.perf_counter()

        vcfg = variant_spec["config"]
        config = TrainConfig(
            window_size=window_size,
            batch_size=vcfg.get("batch_size", 16),
            num_epochs=vcfg.get("num_epochs", dataset_cfg["num_epochs"]),
            lr=vcfg.get("lr", 1e-4),
            l1_lambda=vcfg.get("l1_lambda", 7.5),
            competitive_lambda=vcfg.get("competitive_lambda", 15.0),
            indirect_lambda=vcfg.get("indirect_lambda", 0.0),
            model_type=vcfg.get("model_type", "standard"),
            device=str(device),
            seed=run_idx,
            output_dir=str(output_dir),
        )

        result = train_model(
            spike_data=spike_tensor,
            num_neurons=num_channels,
            config=config,
            criterion=criterion,
            encoder_kwargs=variant_spec["encoder_kwargs"],
            gnn_kwargs=variant_spec["gnn_kwargs"],
            run_id=run_idx,
        )

        encoder, gnn = result["encoder"], result["gnn"]

        # Load best checkpoint for this run
        best_path = output_dir / f"best_model_run_{run_idx}.pt"
        if best_path.exists():
            load_checkpoint(str(best_path), encoder, gnn)

        # Save learned adjacency and plot
        learned_A = extract_adjacency(gnn)
        np.save(output_dir / f"gnn_{run_idx}.npy", learned_A)
        if dataset_name in ('pdcm', 'pdcm_large'):
            plot_pdcm_matrix(learned_A, pop_labels, method_name=f"GNN {run_idx}", save_dir=str(output_dir))
        elif dataset_name == 'mdp':
            plot_mdp_matrix(learned_A, neuron_names, method_name=f"GNN {run_idx}", save_dir=str(output_dir))
        elif dataset_name == 'v1_v2':
            plot_v1_v2_matrix(learned_A, neuron_names, method_name=f"GNN {run_idx}", save_dir=str(output_dir))
        elif dataset_name == 'hc3':
            plot_hc3_matrix(learned_A, neuron_names, method_name=f"GNN {run_idx}", save_dir=str(output_dir))
        else:
            plot_connectivity_matrix(learned_A, neuron_names=neuron_names,
                                     method_name=f"GNN {run_idx}", save_dir=str(output_dir))

        # Ablation test
        encoder.eval()
        gnn.eval()
        val_loader = result["val_loader"]
        val_full = eval_with_scale(encoder, gnn, val_loader, criterion, device, A_scale=1.0)
        val_no_A = eval_with_scale(encoder, gnn, val_loader, criterion, device, A_scale=0.0)
        diff = val_full - val_no_A

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([run_idx, val_full, val_no_A, diff])

        elapsed = time.perf_counter() - t0
        print(f"  val(A)={val_full:.4f}  val(no-A)={val_no_A:.4f}  diff={diff:.4f}  [{elapsed:.1f}s]")


def main():
    parser = argparse.ArgumentParser(description="circuitGNN variant sweep")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_REGISTRY.keys()),
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Datasets to run (default: all)")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()),
                        help="Variants to run (default: all)")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of random-seed runs per (dataset, variant) pair")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Variants: {args.variants}")
    print(f"Runs per config: {args.num_runs}")
    print(f"Results → {RESULTS_BASE}\n")

    for dataset_name in args.datasets:
        for variant_name in args.variants:
            print(f"\n{'='*60}")
            print(f"  Dataset: {dataset_name}   Variant: {variant_name}")
            print(f"{'='*60}")
            run_dataset_variant(
                dataset_name=dataset_name,
                variant_name=variant_name,
                dataset_cfg=DATASET_REGISTRY[dataset_name],
                variant_spec=VARIANTS[variant_name],
                num_runs=args.num_runs,
                device=device,
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()
# http://bactra.org/notebooks/directed-information-and-transfer-entropy.html
# https://www.statlect.com/asymptotic-theory/plug-in-principle
# https://github.com/cbueth/infomeasure/

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))
from benchmarks.standard.correlation import *
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from functools import partial
from circuit_gnn.data import *
from circuit_gnn.utils import *

# hc3 is in-vivo (no ground truth); load_hc3_session needs session-specific
# kwargs but returns the same shape and takes downsample_factor first.
load_hc3 = partial(
    load_hc3_session,
    tar_path='datasets/hc3/ec016.1016.tar.gz',
    topdir='ec016.58',
    cell_csv='datasets/hc3/hc3-cell.csv',
    regions=['CA3', 'DG', 'EC2', 'EC3'],
    quality_filter=True,
    max_time_s=2000,
)

data_map = {'yuto': load_yuto_network, 'arbitrary': load_arbitrary_network, 'pdcm': load_pdcm_network, 'hc3': load_hc3}

native_res_ms = {'yuto': 0.5, 'arbitrary': 1.0, 'pdcm': 1.0, 'hc3': 1.0} # in ms
bin_sizes_ms = [1, 2, 5, 10, 50]

for data in ['hc3']:
    for bin_size_ms in bin_sizes_ms:

        downsample_factor = int(bin_size_ms / native_res_ms[data])
        suffix = f"_{bin_size_ms}ms"
        max_lag = 30

        save_dir = f"datasets/{data}/init"
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n=== {data} | {bin_size_ms} ms bins (downsample_factor={downsample_factor}) ===")

        # Load data
        binned_spikes, true_conn, neuron_names = data_map[data](downsample_factor=downsample_factor)
        num_channels, num_time = binned_spikes.shape
        print(f'{num_channels} channels, {num_time} time points')

        # Correlation
        start_time = time.time()
        ci_correlation, pk_correlation = calculate_correlation(binned_spikes, max_lag=max_lag)
        end_time = time.time()
        print(f"Correlation calculation time: {(end_time - start_time)/60:.2f} minutes")

        np.save(f"{save_dir}/correlation_pk{suffix}.npy", pk_correlation)
        np.save(f"{save_dir}/correlation_ci{suffix}.npy", ci_correlation)
        plot_connectivity_matrix(pk_correlation, neuron_names, f"Correlation PK {suffix}", save_dir=save_dir, plot_val=False)
        plot_connectivity_matrix(ci_correlation, neuron_names, f"Correlation CI {suffix}", save_dir=save_dir, plot_val=False)
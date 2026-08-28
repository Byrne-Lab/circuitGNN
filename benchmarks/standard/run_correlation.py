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
from circuit_gnn.data import *
from circuit_gnn.utils import *

data_map = {'OC_1': load_OC_1_network, 'OC_2': load_OC_2_network,
            'control': load_control_network,
            'arbitrary': load_arbitrary_network, 'pdcm': load_pdcm_network}

native_res_ms = {'OC_1': 0.5, 'OC_2': 0.5, 'control': 0.5, 'arbitrary': 1.0, 'pdcm': 1.0} # in ms
bin_sizes_ms = [1, 2, 5, 10, 50]

for data in ['OC_1', 'OC_2', 'control', 'pdcm']:

    for bin_size_ms in bin_sizes_ms:

        downsample_factor = int(bin_size_ms / native_res_ms[data])
        suffix = f"_{bin_size_ms}ms"
        max_lag = 30

        save_dir = f"datasets/{data}/baseline"
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n=== {data} | {bin_size_ms} ms bins (downsample_factor={downsample_factor}) ===")

        # Load data
        binned_spikes, true_conn, neuron_names = data_map[data](downsample_factor=downsample_factor)
        num_channels, num_time = binned_spikes.shape
        print(f'{num_channels} channels, {num_time} time points')

        # Correlation
        start_time = time.time()
        _, pk_correlation = calculate_correlation(binned_spikes, max_lag=max_lag)
        end_time = time.time()
        print(f"Correlation calculation time: {(end_time - start_time)/60:.2f} minutes")

        np.save(f"{save_dir}/correlation_pk{suffix}.npy", pk_correlation)
        plot_connectivity_matrix(pk_correlation, neuron_names, f"Correlation PK {suffix}", save_dir=save_dir, plot_val=False)
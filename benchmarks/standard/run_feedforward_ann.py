import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))
from benchmarks.standard.feedforward_ann import calculate_frolov_connectivity
import numpy as np
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

EMBED_DIM    = 1
TAU          = 10
SMOOTH_SIGMA = 20.0

for data in ['hc3']:

    if data == 'yuto':
        downsample_factor = 20   # 10 ms bins
    elif data == 'arbitrary':
        downsample_factor = 10   # 10 ms bins
    elif data == 'pdcm':
        downsample_factor = 1    # 1 ms bins
    elif data == 'hc3':
        downsample_factor = 10   # 10 ms bins (native resolution 1 ms)

    save_dir = f"datasets/{data}/init"
    os.makedirs(save_dir, exist_ok=True)

    binned_spikes, true_conn, neuron_names = data_map[data](downsample_factor=downsample_factor)
    num_channels, num_time = binned_spikes.shape
    print(f'{data}: {num_channels} channels, {num_time} time points')

    start_time = time.time()
    r2_matrix, conn_matrix = calculate_frolov_connectivity(
        binned_spikes,
        embed_dim=EMBED_DIM,
        tau=TAU,
        smooth_sigma=SMOOTH_SIGMA,
    )
    elapsed = time.time() - start_time
    print(f"Frolov ANN calculation time: {elapsed/60:.2f} minutes")

    # np.save(f"{save_dir}/frolov_r2.npy", r2_matrix)
    np.save(f"{save_dir}/frolov_conn.npy", conn_matrix)
    # plot_connectivity_matrix(r2_matrix, neuron_names, "Frolov ANN R²", save_dir=save_dir, plot_val=False)
    plot_connectivity_matrix(conn_matrix.astype(float), neuron_names, "Frolov ANN Connections", save_dir=save_dir, plot_val=False)
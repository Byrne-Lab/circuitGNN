import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))
from benchmarks.standard.glm import calculate_glm
from benchmarks.standard.utils import threshold_top_k
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

data_map = {
    'yuto': load_yuto_network,
    'arbitrary': load_arbitrary_network,
    'pdcm': load_pdcm_network,
    'hc3': load_hc3,
}


for data in ['hc3']:

    if data == 'yuto':
        downsample_factor = 2  # 1 ms
    elif data == 'arbitrary':
        downsample_factor = 1  # 1 ms
    elif data == 'pdcm':
        downsample_factor = 1  # 1 ms
    elif data == 'hc3':
        downsample_factor = 1  # 1 ms (native resolution)

    save_dir = f"datasets/{data}/init"
    os.makedirs(save_dir, exist_ok=True)

    binned_spikes, true_conn, neuron_names = data_map[data](downsample_factor=downsample_factor)
    num_channels, num_time = binned_spikes.shape
    print(f'\n=== {data}: {num_channels} channels, {num_time} time points ===')

    start = time.time()
    glm_conn = calculate_glm(binned_spikes, dt_ms=1.0)
    print(f"GLM time: {(time.time() - start) / 60:.2f} min")

    glm_threshold = threshold_top_k(glm_conn, true_conn)

    np.save(f"{save_dir}/glm.npy", glm_conn)
    np.save(f"{save_dir}/glm_threshold.npy", glm_threshold)
    plot_connectivity_matrix(glm_conn, neuron_names, "GLM", save_dir=save_dir, plot_val=False)
    plot_connectivity_matrix(glm_threshold, neuron_names, "GLM_threshold", save_dir=save_dir, plot_val=False)
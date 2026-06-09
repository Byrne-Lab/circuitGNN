import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))
from benchmarks.standard.glm import calculate_glm
import numpy as np
import os
import time
from circuit_gnn.data import *
from circuit_gnn.utils import *

data_map = {
    'yuto': load_yuto_network,
    'arbitrary': load_arbitrary_network,
    'pdcm': load_pdcm_network,
}

for data in ['yuto', 'arbitrary', 'pdcm']:

    if data == 'yuto':
        downsample_factor = 2  # 1 ms
    else:
        downsample_factor = 1  # 1 ms

    save_dir = f"datasets/{data}/init"
    os.makedirs(save_dir, exist_ok=True)

    binned_spikes, true_conn, neuron_names = data_map[data](downsample_factor=downsample_factor)
    num_channels, num_time = binned_spikes.shape
    print(f'\n=== {data}: {num_channels} channels, {num_time} time points ===')

    start = time.time()
    glm_conn = calculate_glm(binned_spikes, dt_ms=1.0)
    print(f"GLM time: {(time.time() - start) / 60:.2f} min")

    np.save(f"{save_dir}/glm.npy", glm_conn)
    plot_connectivity_matrix(glm_conn, neuron_names, "GLM (Gerhard 2013)", save_dir=save_dir, plot_val=False)

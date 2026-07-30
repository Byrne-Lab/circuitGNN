import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import h5py
import numpy as np
import os
import seaborn as sns
from circuit_gnn.utils import plot_connectivity_matrix, plot_pdcm_matrix

RED_BLUE = sns.diverging_palette(240, 10, as_cmap=True)

os.chdir('C:/Users/agupta17/Documents/GitHub/circuitGNN/datasets')
data_name = 'pdcm'

data = f"{data_name}/baseline/ctm_di_raw_connectivity.mat"

def threshold_di(C, min_consecutive=4, threshold=0.01):
    n_bins, n, _ = C.shape
    C_final = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            curve = C[:, i, j]
            consecutive = 0
            has_plateau = False
            for val in np.abs(curve):
                if val >= threshold:
                    consecutive += 1
                    if consecutive >= min_consecutive:
                        has_plateau = True
                        break
                else:
                    consecutive = 0
            if has_plateau:
                C_final[i, j] = np.mean(curve[curve != 0])
    return C_final

with h5py.File(data, "r") as f:
    C = np.array(f["conn_3d"])
    C = C.transpose(0, 2, 1)
    print(f"Raw connectivity shape: {C.shape}")

C_thresh = threshold_di(C)
print(f"Thresholded connectivity shape: {C_thresh.shape}")
print(f"Non-zero connections: {np.count_nonzero(C_thresh)}")

save_dir = f"{data_name}/baseline"
np.save(f"{save_dir}/CTM_DI.npy", C_thresh)

if data_name in ('pdcm', 'pdcm_large'):
    pop_labels = np.load(f'{data_name}/pop_labels.npy')
    plot_pdcm_matrix(C_thresh, pop_labels, method_name='CTM-DI', save_dir=save_dir)
elif data_name in ('OC_2', 'small_circuit_A', 'small_circuit_B'):
    with open(f'{data_name}/spikes.head') as f:
        neuron_names = [line.split('[')[1].split('.')[0] for line in f if line.startswith('V[')]
    plot_connectivity_matrix(C_thresh, neuron_names, method_name='CTM-DI',
                             cmap=RED_BLUE, save_dir=save_dir, plot_val=False)
else:
    neuron_names = [f'Neuron {i+1}' for i in range(C_thresh.shape[0])]
    plot_connectivity_matrix(C_thresh, neuron_names, method_name='CTM-DI',
                             cmap=RED_BLUE, save_dir=save_dir, plot_val=False)

import copy
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import os
os.chdir('C:/Users/agupta17/Documents/GitHub/circuitGNN/datasets')
data_name = 'yuto'

data = f"{data_name}/init/ctm_di_raw_connectivity.mat"
with open('C:/Users/agupta17/Documents/GitHub/circuitGNN/datasets/yuto/spikes.head') as f:
    neuron_names = [line.split('[')[1].split('.')[0] for line in f if line.startswith('V[')]

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
    print(f"Raw connectivity shape: {C.shape}")  # should be (n_bins, n_neurons, n_neurons)


C_thresh = threshold_di(C)
print(C)
print(f"Thresholded connectivity shape: {C_thresh.shape}")
print(f"Non-zero connections: {np.count_nonzero(C_thresh)}")

# Plot
vmin = min(C_thresh.min(), -1e-6)
vmax = max(C_thresh.max(),  1e-6)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = sns.diverging_palette(240, 10, as_cmap=True)
cmap = copy.copy(cmap)
cmap.set_bad(color='white')
masked = np.ma.masked_where(C_thresh == 0, C_thresh)
annot_labels = np.where(C_thresh == 0, '', np.vectorize(lambda v: f"{v:.3f}")(C_thresh))

plt.figure(figsize=(10, 8))
sns.heatmap(
    masked,
    xticklabels=neuron_names,
    yticklabels=neuron_names,
    cmap=cmap,
    norm=norm,
    cbar_kws={'label': 'Connectivity Strength'},
    square=True,
    annot=annot_labels,
    fmt='',
)
plt.title("CTM-DI Connectivity Matrix")
plt.xlabel("Postsynaptic neuron")
plt.ylabel("Presynaptic neuron")
plt.tight_layout()
plt.savefig(f"{data_name}/init/CTM_DI.pdf", format='pdf')
np.save(f"{data_name}/init/CTM_DI.npy", C_thresh)
plt.close()
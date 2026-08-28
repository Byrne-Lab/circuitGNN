import os
import sys

# put the `src` dir (parent of this package) on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_gnn.utils import *
from circuit_gnn.data import *
import pandas as pd

data = 'arbitrary'

num_runs = 50
ablation_threshold = 0.10 # removes non-optimal runs based on ablation test
sign_bin_ms = 20
_native_ms = {'OC_1': 0.5, 'OC_2': 0.5, 'control': 0.5,
              'small_circuit_A': 0.5, 'small_circuit_B': 0.5}
sign_bin = round(sign_bin_ms / _native_ms.get(data, 1.0))

# CREATE FOLDERS
print(f"\n{data} Evaluation")
results_loc = f'results/{data}'
ablation_loc = f'{results_loc}/ablation.csv'

# PRINT SETTINGS
print(f"Data: {data}")
print(f"Num runs: {num_runs}")
print(f"Ablation threshold: {ablation_threshold}")

# LOAD DATA
data_map = {'OC_1': load_OC_1_network, 'OC_2': load_OC_2_network, 'control': load_control_network,
            'arbitrary': load_arbitrary_network, 'pdcm': load_pdcm_network,
            'arbitrary_modification1': load_arbitrary_modification1_network,
            'arbitrary_modification2': load_arbitrary_modification2_network,
            'small_circuit_A': load_small_circuit_A_network, 'small_circuit_B': load_small_circuit_B_network}

binned_spikes, true_conn, neuron_names = data_map[data](downsample_factor=sign_bin)

if data == 'pdcm':
    pop_labels = np.load('datasets/pdcm/pop_labels.npy')
    def do_plot(matrix, method_name):
        plot_pdcm_matrix(matrix, pop_labels, method_name=method_name, save_dir=results_loc)
else:
    def do_plot(matrix, method_name):
        plot_connectivity_matrix(matrix, neuron_names=neuron_names, method_name=method_name, save_dir=results_loc, plot_val=True, vmin=-0.015, vmax=0.010)
ablation_csv = pd.read_csv(ablation_loc)

# RUN EVALUATION
adjacency_matrices = []
passed_run_ids = []
num_failed_runs = 0
for run_idx in range(1, num_runs+1):
    ablation_row = ablation_csv[ablation_csv['run_id'] == run_idx]
    if ablation_row.difference.values[0] > -ablation_threshold:
        num_failed_runs+=1
        print(f'Skipping run {run_idx} for failing the ablation test')
        continue
    fname = f"{results_loc}/gnn_{run_idx}.npy"
    A = np.load(fname)
    adjacency_matrices.append(A)
    passed_run_ids.append(run_idx)
print(f'Skipped {num_failed_runs} failed runs')

# Regular mean
before = [np.sign(A.sum()) for A in adjacency_matrices]
adjacency_matrices = align_matrices(adjacency_matrices, bias=None, binned_spikes=binned_spikes)
after = [np.sign(A.sum()) for A in adjacency_matrices]
flipped = [passed_run_ids[i] for i,(b,a) in enumerate(zip(before,after)) if b!=a]
print(f"Flipped runs: {flipped}")

adjacency_matrices = np.stack(adjacency_matrices, axis=0)
consistency_mask = get_consistent_connections(adjacency_matrices, threshold=0.8)
mean_A = adjacency_matrices.mean(axis=0)

mean_A = np.round(mean_A, 3)
consistent_mean_A = mean_A * consistency_mask # apply consistency mask

np.save(f'{results_loc}/mean.npy', mean_A)
np.save(f'{results_loc}/mean_consistent.npy', consistent_mean_A)

do_plot(mean_A, 'mean')
res = calculate_fit(true_conn, mean_A)
print(f"\nMean: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, Signed F1={res.signed_f1:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")

do_plot(consistent_mean_A, 'mean_consistent')
res = calculate_fit(true_conn, consistent_mean_A)
print(f"\nConsistent Mean: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, Signed F1={res.signed_f1:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")

import os
import sys

# put the `src` dir (parent of this package) on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_gnn.utils import *
from circuit_gnn.data import *
import pandas as pd
from functools import partial

# SET PARAMETERS
data = 'pdcm_0p002_120s' 
gap = 0.20 # only useful for yuto
step = 1 # only useful for yuto

num_runs = 20
ablation_threshold = 0.10 # removes non-optimal runs based on ablation test
bias = None
sign_bin_ms = 20
_native_ms = {'yuto': 0.5, 'small_circuit_A': 0.5, 'small_circuit_B': 0.5}
sign_bin = round(sign_bin_ms / _native_ms.get(data, 1.0))
plot = False # plot metrics or not

# CREATE FOLDERS
print(f"\n{data} Evaluation")
results_loc = f'results/{data}'
ablation_loc = f'{results_loc}/ablation.csv'

# PRINT SETTINGS    
print(f"Data: {data}")
print(f"Num runs: {num_runs}")
print(f"Ablation threshold: {ablation_threshold}")

# LOAD DATA
# hc3_gor01 / hc3_pin01 are dedicated single-session loaders (config baked in);
# set `data` above to one of these names to evaluate that session.
data_map = {'yuto': load_yuto_network, 'arbitrary': load_arbitrary_network, 'pdcm': load_pdcm_network, 'mdp': load_mdp_network,
            'small_circuit_A': load_small_circuit_A_network, 'small_circuit_B': load_small_circuit_B_network,
            'hc3': load_hc3_ec016, 'hc3_gor01': load_hc3_gor01, 'hc3_pin01': load_hc3_pin01, 'hc3_ec014': load_hc3_ec014,
            # ec015.03 behavioural stages (same neuron set across stages -> comparable A)
            'hc3_ec015_sleep': partial(load_hc3_ec015, stage='sleep'),
            'hc3_ec015_wheel': partial(load_hc3_ec015, stage='wheel'),
            'hc3_ec015_midSquare': partial(load_hc3_ec015, stage='midSquare')}

# PDCM scaling-sweep names, e.g. 'pdcm_0p002_60s' -> size='0p002', max_time_s=60.
pdcm_size = None
_m = re.match(r'pdcm_(\dp\d+)(?:_(\d+)s)?$', data)
if _m:
    pdcm_size = _m.group(1)
    _pdcm_kwargs = {'size': pdcm_size, 'plot': False}
    if _m.group(2):
        _pdcm_kwargs['max_time_s'] = int(_m.group(2))
    data_map[data] = partial(load_pdcm_network, **_pdcm_kwargs)

binned_spikes, true_conn, neuron_names = data_map[data](gap=gap, step=step, downsample_factor=sign_bin)

# Plot function: pdcm/v1 use population-boundary format, others use standard
if data == 'pdcm' or pdcm_size is not None:
    _pdcm_dir = 'datasets/pdcm' if pdcm_size is None else f'datasets/pdcm_{pdcm_size}'
    pop_labels = np.load(f'{_pdcm_dir}/pop_labels.npy')
    def do_plot(matrix, method_name):
        plot_pdcm_matrix(matrix, pop_labels, method_name=method_name, save_dir=results_loc)
elif data == 'mdp':
    def do_plot(matrix, method_name):
        plot_mdp_matrix(matrix, neuron_names, method_name=method_name, save_dir=results_loc)
elif data.startswith('hc3'):
    def do_plot(matrix, method_name):
        plot_hc3_matrix(matrix, neuron_names, method_name=method_name, save_dir=results_loc)
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
    if np.abs(ablation_row.difference.values[0]) < ablation_threshold:
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
adjacency_matrices = align_matrices(adjacency_matrices, bias=bias, binned_spikes=binned_spikes)
after = [np.sign(A.sum()) for A in adjacency_matrices]
flipped = [passed_run_ids[i] for i,(b,a) in enumerate(zip(before,after)) if b!=a]
print(f"Flipped runs: {flipped}")

adjacency_matrices = np.stack(adjacency_matrices, axis=0)
consistency_mask = get_consistent_connections(adjacency_matrices, threshold=0.8)
mean_A = adjacency_matrices.mean(axis=0)

mean_A = np.round(mean_A, 3)
consistent_mean_A = mean_A * consistency_mask # apply consistency mask

do_plot(mean_A, 'mean')
res = calculate_fit(true_conn, mean_A, plot=plot)
print(f"\nMean: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, Signed F1={res.signed_f1:.2f}, MCC={res.mcc_unsigned:.2f}, Signed MCC={res.mcc_signed:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")

do_plot(consistent_mean_A, 'mean_consistent')
res = calculate_fit(true_conn, consistent_mean_A, plot=plot)
print(f"\nConsistent Mean: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, Signed F1={res.signed_f1:.2f}, MCC={res.mcc_unsigned:.2f}, Signed MCC={res.mcc_signed:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")
import os
import sys

# put the `src` dir (parent of this package) on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_gnn.utils import *
from circuit_gnn.data import *
import pandas as pd
from functools import partial

# hc3 is in-vivo (no ground truth); load_hc3_session needs session-specific
# kwargs but returns the same (binned_spikes, true_conn, neuron_names) shape.
# Same session/window as the standard baselines so the neuron set aligns.
load_hc3 = partial(
    load_hc3_session,
    tar_path='datasets/hc3/ec016.1016.tar.gz',
    topdir='ec016.58',
    cell_csv='datasets/hc3/hc3-cell.csv',
    regions=['CA3', 'DG', 'EC2', 'EC3'],
    quality_filter=True,
    max_time_s=2000,
)

# SET PARAMETERS
data = 'hc3_pin01' # yuto / arbitrary / pdcm / mdp / hc3 / hc3_gor01 / hc3_pin01 / hc3_ec014
gap = 0.20 # only useful for yuto
step = 1 # only useful for yuto

init = False
signed = False
num_runs = 5
# KEY PARAMETERS
ablation_threshold = 0.20 # removes non-optimal runs based on ablation test
excitatory = True # excitatory connections dominate (True) or inhibitory connections dominate (False) 
plot = False # plot metrics or not

# CREATE FOLDERS
print(f"\n{data} Evaluation")
data_loc = f'datasets/{data}/init'
results_loc = f'results/{data}/no_init' if not init else(f'results/{data}/unsigned' if not signed else f'results/{data}/signed')
ablation_loc = f'{results_loc}/ablation.csv'

# PRINT SETTINGS    
print(f"Data: {data}")
print(f"Init: {init}")
print(f"Signed Init: {signed}")
print(f"Num runs: {num_runs}")
print(f"Ablation threshold: {ablation_threshold}")

# LOAD DATA
# hc3_gor01 / hc3_pin01 are dedicated single-session loaders (config baked in);
# set `data` above to one of these names to evaluate that session.
data_map = {'yuto': load_yuto_network, 'arbitrary': load_arbitrary_network, 'pdcm': load_pdcm_network, 'mdp': load_mdp_network,
            'hc3': load_hc3, 'hc3_gor01': load_hc3_gor01, 'hc3_pin01': load_hc3_pin01, 'hc3_ec014': load_hc3_ec014}
binned_spikes, true_conn, neuron_names = data_map[data](gap=gap, step=step)

# Plot function: pdcm/v1 use population-boundary format, others use standard
if data == 'pdcm':
    pop_labels = np.load(f'datasets/pdcm/pop_labels.npy')
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
        plot_connectivity_matrix(matrix, neuron_names=neuron_names, method_name=method_name, save_dir=results_loc, plot_val=True)
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
adjacency_matrices = align_matrices(adjacency_matrices, excitatory=excitatory)
after = [np.sign(A.sum()) for A in adjacency_matrices]
flipped = [passed_run_ids[i] for i,(b,a) in enumerate(zip(before,after)) if b!=a]
print(f"Flipped runs: {flipped}")


# adjacency_matrices = align_matrices(adjacency_matrices, excitatory=excitatory)
adjacency_matrices = np.stack(adjacency_matrices, axis=0)
consistency_mask = get_consistent_connections(adjacency_matrices, threshold=0.8)
mean_A = adjacency_matrices.mean(axis=0)

mean_A = np.round(mean_A, 3)

# Consistent mean
consistent_mean_A = mean_A * consistency_mask # apply consistency mask
# Top K consistent mean
consistent_mean_A_top = get_top_k(consistent_mean_A, 0.1)
# GDI Signed mean
gdi_sign, di_sign = sign_inference(binned_spikes.T, 3) # obtain sign
signed_mean_A = np.abs(mean_A) * gdi_sign

do_plot(mean_A, 'mean')
res = calculate_fit(true_conn, mean_A, plot=plot)
print(f"\nMean: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, Signed F1={res.signed_f1:.2f}, MCC={res.mcc_unsigned:.2f}, Signed MCC={res.mcc_signed:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")

do_plot(consistent_mean_A, 'mean_consistent')
res = calculate_fit(true_conn, consistent_mean_A, plot=plot)
print(f"\nConsistent Mean: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, Signed F1={res.signed_f1:.2f}, MCC={res.mcc_unsigned:.2f}, Signed MCC={res.mcc_signed:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")

do_plot(consistent_mean_A_top, 'mean_consistent_topk')
res = calculate_fit(true_conn, consistent_mean_A_top, plot=plot)
print(f"\nConsistent Mean Top-K: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, Signed F1={res.signed_f1:.2f}, MCC={res.mcc_unsigned:.2f}, Signed MCC={res.mcc_signed:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")

do_plot(signed_mean_A, 'mean_signed_Young_et_al')
res = calculate_fit(true_conn, signed_mean_A, plot=plot)
print(f"\nYoung et al. Signed Mean: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, Signed F1={res.signed_f1:.2f}, MCC={res.mcc_unsigned:.2f}, Signed MCC={res.mcc_signed:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")

# Baseline: CTM_DI (Cai et al.)
A_init_path_cai = os.path.join(f"{data_loc}/CTM_DI.npy")
A_init_cai = np.load(A_init_path_cai)
do_plot(A_init_cai, 'mean_CTM-DI')
res = calculate_fit(true_conn, A_init_cai, plot=plot)
print(f"\n Cai et al: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, Signed F1={res.signed_f1:.2f}, MCC={res.mcc_unsigned:.2f}, Signed MCC={res.mcc_signed:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")

# Baseline: Correlation
A_init_path_corr = os.path.join(f"{data_loc}/correlation.npy")
A_init_corr = np.load(A_init_path_corr)
A_init_corr = np.round(A_init_corr, 3)

do_plot(A_init_corr, 'correlation')
res = calculate_fit(true_conn, A_init_corr, plot=plot)
print(f"\nCorrelation: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, Signed F1={res.signed_f1:.2f}, MCC={res.mcc_unsigned:.2f}, Signed MCC={res.mcc_signed:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")

# Baseline: Signed Correlation
A_init_path_corr = os.path.join(f"{data_loc}/signed_correlation.npy")
A_init_corr = np.load(A_init_path_corr)
A_init_corr = np.round(A_init_corr, 3)

do_plot(A_init_corr, 'signed_correlation')
res = calculate_fit(true_conn, A_init_corr, plot=plot)
print(f"\nSigned Correlation: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, Signed F1={res.signed_f1:.2f}, MCC={res.mcc_unsigned:.2f}, Signed MCC={res.mcc_signed:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")

# Baseline: Mutual information
A_init_path_mi = os.path.join(f"{data_loc}/signed_mutual_information.npy")
A_init_mi = np.load(A_init_path_mi)
A_init_mi = np.round(A_init_mi, 3)

do_plot(A_init_mi, 'signed_mutual_information')
res = calculate_fit(true_conn, A_init_mi, plot=plot)
print(f"\nSigned Mutual Information: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, Signed F1={res.signed_f1:.2f}, MCC={res.mcc_unsigned:.2f}, Signed MCC={res.mcc_signed:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")

# Baseline: Transfer entropy
A_init_path_te = os.path.join(f"{data_loc}/signed_transfer_entropy.npy")
A_init_te = np.load(A_init_path_te)
A_init_te = np.round(A_init_te, 3)

do_plot(A_init_te, 'signed_transfer_entropy')
res = calculate_fit(true_conn, A_init_te, plot=plot)
print(f"\nSigned Transfer Entropy: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, Signed F1={res.signed_f1:.2f}, MCC={res.mcc_unsigned:.2f}, Signed MCC={res.mcc_signed:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")
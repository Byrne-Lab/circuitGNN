from utils import *
from data import *
import pandas as pd

# SET PARAMETERS
data = 'pdcm'
gap = 0.20 # only useful for yuto

init = False
signed = False
num_runs = 3
# KEY PARAMETERS
ablation_threshold = 0.25 # removes non-optimal runs based on ablation test
excitatory = True # Excitatory connections dominate (True) or inhibitory connections dominate (False) 

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
data_map = {'yuto': load_yuto_network, 'arbitrary': load_arbitrary_network, 'pdcm': load_pdcm_network}
binned_spikes, true_conn, neuron_names = data_map[data](gap=gap)

# Plot function: pdcm uses population-boundary format, others use standard
if data == 'pdcm':
    pop_labels = np.load(f'datasets/pdcm/pop_labels.npy')
    def do_plot(matrix, method_name):
        plot_pdcm_matrix(matrix, pop_labels, method_name=method_name, save_dir=results_loc)
else:
    def do_plot(matrix, method_name):
        plot_connectivity_matrix(matrix, neuron_names=neuron_names, method_name=method_name, save_dir=results_loc, plot_val=False)
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
cosine_mean, pearson_mean, spearman_mean, overall_precision_mean, overall_recall_mean, overall_f1_mean, signed_precision_mean, signed_recall_mean, signed_f1_mean = calculate_fit(true_conn, mean_A)
print(f"\nMean: Cosine={cosine_mean:.2f}, Pearson={pearson_mean:.2f}, Spearman={spearman_mean:.2f}, Overall Precision={overall_precision_mean:.2f}, Overall Recall={overall_recall_mean:.2f}, Overall F1={overall_f1_mean:.2f}, Signed Precision={signed_precision_mean:.2f}, Signed Recall={signed_recall_mean:.2f}, Signed F1={signed_f1_mean:.2f}")

do_plot(consistent_mean_A, 'mean_consistent')
cosine_mean, pearson_mean, spearman_mean, overall_precision_consistent, overall_recall_consistent, overall_f1_consistent, signed_precision_consistent, signed_recall_consistent, signed_f1_consistent = calculate_fit(true_conn, consistent_mean_A)
print(f"\nConsistent Mean: Cosine={cosine_mean:.2f}, Pearson={pearson_mean:.2f}, Spearman={spearman_mean:.2f}, Overall Precision={overall_precision_consistent:.2f}, Overall Recall={overall_recall_consistent:.2f}, Overall F1={overall_f1_consistent:.2f}, Signed Precision={signed_precision_consistent:.2f}, Signed Recall={signed_recall_consistent:.2f}, Signed F1={signed_f1_consistent:.2f}")

do_plot(consistent_mean_A_top, 'mean_consistent_topk')
cosine_mean, pearson_mean, spearman_mean, overall_precision_topk, overall_recall_topk, overall_f1_topk, signed_precision_topk, signed_recall_topk, signed_f1_topk = calculate_fit(true_conn, consistent_mean_A_top)
print(f"\nConsistent Mean Top-K: Cosine={cosine_mean:.2f}, Pearson={pearson_mean:.2f}, Spearman={spearman_mean:.2f}, Overall Precision={overall_precision_topk:.2f}, Overall Recall={overall_recall_topk:.2f}, Overall F1={overall_f1_topk:.2f}, Signed Precision={signed_precision_topk:.2f}, Signed Recall={signed_recall_topk:.2f}, Signed F1={signed_f1_topk:.2f}")

do_plot(signed_mean_A, 'mean_signed_Young_et_al')
cosine_mean, pearson_mean, spearman_mean, overall_precision_topk, overall_recall_topk, overall_f1_topk, signed_precision_topk, signed_recall_topk, signed_f1_topk = calculate_fit(true_conn, signed_mean_A)
print(f"\nYoung et al. Signed Mean: Cosine={cosine_mean:.2f}, Pearson={pearson_mean:.2f}, Spearman={spearman_mean:.2f}, Overall Precision={overall_precision_topk:.2f}, Overall Recall={overall_recall_topk:.2f}, Overall F1={overall_f1_topk:.2f}, Signed Precision={signed_precision_topk:.2f}, Signed Recall={signed_recall_topk:.2f}, Signed F1={signed_f1_topk:.2f}")

# Baseline: Cai et al
A_init_path_cai = os.path.join(f"{data_loc}/CTM-DI.npy")
A_init_cai = np.load(A_init_path_cai)
do_plot(A_init_cai, 'mean_CTM-DI')
cosine_cai, pearson_cai, spearman_cai, overall_precision_cai, overall_recall_cai, overall_f1_cai, signed_precision_cai, signed_recall_cai, signed_f1_cai = calculate_fit(true_conn, A_init_cai)
print(f"\n Cai et al: Cosine={cosine_cai:.2f}, Pearson={pearson_cai:.2f}, Spearman={spearman_cai:.2f}, Overall Precision={overall_precision_cai:.2f}, Overall Recall={overall_recall_cai:.2f}, Overall F1={overall_f1_cai:.2f}, Signed Precision={signed_precision_cai:.2f}, Signed Recall={signed_recall_cai:.2f}, Signed F1={signed_f1_cai:.2f}")

# Baseline: signed_correlation
A_init_path_corr = os.path.join(f"{data_loc}/signed_correlation.npy")
A_init_corr = np.load(A_init_path_corr)
A_init_corr = A_init_corr.T
do_plot(A_init_corr, 'signed_correlation')
cosine_corr, pearson_corr, spearman_corr, overall_precision_corr, overall_recall_corr, overall_f1_corr, signed_precision_corr, signed_recall_corr, signed_f1_corr = calculate_fit(true_conn, A_init_corr)
print(f"\nSigned Correlation: Cosine={cosine_corr:.2f}, Pearson={pearson_corr:.2f}, Spearman={spearman_corr:.2f}, Overall Precision={overall_precision_corr:.2f}, Overall Recall={overall_recall_corr:.2f}, Overall F1={overall_f1_corr:.2f}, Signed Precision={signed_precision_corr:.2f}, Signed Recall={signed_recall_corr:.2f}, Signed F1={signed_f1_corr:.2f}")

# Baseline: signed_transfer_entropy
A_init_path_te = os.path.join(f"{data_loc}/signed_transfer_entropy.npy")
A_init_te = np.load(A_init_path_te)
A_init_te = A_init_te.T
do_plot(A_init_te, 'signed_transfer_entropy')
cosine_te, pearson_te, spearman_te, overall_precision_te, overall_recall_te, overall_f1_te, signed_precision_te, signed_recall_te, signed_f1_te = calculate_fit(true_conn, A_init_te)
print(f"\nSigned Transfer Entropy: Cosine={cosine_te:.2f}, Pearson={pearson_te:.2f}, Spearman={spearman_te:.2f}, Overall Precision={overall_precision_te:.2f}, Overall Recall={overall_recall_te:.2f}, Overall F1={overall_f1_te:.2f}, Signed Precision={signed_precision_te:.2f}, Signed Recall={signed_recall_te:.2f}, Signed F1={signed_f1_te:.2f}")
import os
import sys

# put the `src` dir (parent of this package) on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_gnn.utils import *
from circuit_gnn.data import *
import pandas as pd

# SET PARAMETERS
# doubles as the results dir name, so it has to match the `data` arg of the run
# script that produced it:
#   'allen'            hippocampal prep_771160300      (run_allen.py)
#   'allen_750749662'  visual hierarchy                (run_allen.py)
#   'th2_<session>_<epoch>'  one th-2 PoSub session    (run_th2.py)
data = 'th2_A3716-201015b_wake1'

num_runs = None
ablation_threshold = 0.01 # removes non-optimal runs based on ablation test
sign_bin_ms = 20
_native_ms = {'allen': 1.0, 'allen_750749662': 1.0}
sign_bin = round(sign_bin_ms / _native_ms.get(data, 1.0))   # th-2 is 1.0 ms too

# CREATE FOLDERS
print(f"\n{data} Aggregation (no ground truth)")
results_loc = f'results/{data}'
ablation_loc = f'{results_loc}/ablation.csv'

# PRINT SETTINGS
print(f"Data: {data}")
print(f"Num runs: {num_runs}")
print(f"Ablation threshold: {ablation_threshold}")

# LOAD DATA
data_map = {'allen': load_allen_network,
            'allen_750749662': load_allen_network}

loader_kwargs = {
    'allen': {'data_dir': 'datasets/allen/prep_771160300',
              'start_time_s': 600, 'max_time_s': 600},
    'allen_750749662': {'data_dir': 'datasets/allen/prep_750749662_natural_movie_three',
                        'start_time_s': 0, 'max_time_s': 600},
}

# th-2 dirs are named th2_<session>_<epoch>, so the loader args come from the
# name rather than an entry per session. max_time_s must match run_th2.py.
if data.startswith('th2_'):
    _session, _epoch = data[len('th2_'):].rsplit('_', 1)
    data_map[data] = load_th2_network
    loader_kwargs[data] = {'data_dir': f'datasets/th2/prep_{_session}_{_epoch}',
                           'start_time_s': 0, 'max_time_s': 600}

binned_spikes, _, neuron_names = data_map[data](downsample_factor=sign_bin,
                                                **loader_kwargs.get(data, {}))

# group label per neuron; allen names are '<region>_<probe>_u<unit_id>',
# th-2 names are '<cellclass>_pos_u<unit>' -- either way the group is token 0
group_labels = [str(n).split('_')[0] for n in neuron_names]
groups = list(dict.fromkeys(group_labels))  # first-appearance order (loaders group by class/region)
print(f"Groups: {groups}")

def do_plot(matrix, method_name):
    plot_region_matrix(matrix, group_labels, groups, method_name=method_name, save_dir=results_loc)

ablation_csv = pd.read_csv(ablation_loc)

# RUN AGGREGATION
adjacency_matrices = []
passed_run_ids = []
num_failed_runs = 0
run_ids = list(ablation_csv['run_id']) if num_runs is None else list(range(1, num_runs+1))
for run_idx in run_ids:
    fname = f"{results_loc}/gnn_{run_idx}.npy"
    ablation_row = ablation_csv[ablation_csv['run_id'] == run_idx]
    if ablation_row.empty or not os.path.exists(fname):
        print(f'Skipping run {run_idx}: no ablation row or no saved matrix')
        continue
    if ablation_row.difference.values[0] > -ablation_threshold:
        num_failed_runs+=1
        print(f'Skipping run {run_idx} for failing the ablation test')
        continue
    A = np.load(fname)
    adjacency_matrices.append(A)
    passed_run_ids.append(run_idx)
print(f'Skipped {num_failed_runs} failed runs; aggregating runs {passed_run_ids}')

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
do_plot(consistent_mean_A, 'mean_consistent')

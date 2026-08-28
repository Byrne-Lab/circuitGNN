addpath('C:/Users/agupta17/Documents/GitHub/npy-matlab/npy-matlab');
repo_dir = 'C:/Users/agupta17/Documents/GitHub/circuitGNN';
data_dir = [repo_dir '/datasets/allen/prep_793224716'];
save_dir = [repo_dir '/datasets/allen/baseline'];

% Same window as scripts/run_allen.py and benchmarks/standard/run_spycon.py
max_time_ms = 600 * 1000;

% Load spike data
spkt   = readNPY([data_dir '/spkt.npy']);   % spike times in ms
spkid  = readNPY([data_dir '/spkid.npy']);  % neuron ID for each spike (0-indexed)

% Unit count comes from the full record, before truncation: a unit that happens
% to be silent in the first 600 s must still get its own (empty) column, or the
% matrix stops lining up with pop_labels.npy on the Python side.
n_neurons = double(max(spkid)) + 1;

keep  = spkt < max_time_ms;
spkt  = spkt(keep);
spkid = spkid(keep);

% Floor to 1 ms integer bin (equivalent to downsample_factor=1 in Python), then
% shift by +1: spiketime2train discards spikes at t == 0 because 0 doubles as
% the zero-padding sentinel in spike_matrix.
spkt = floor(spkt) + 1;

% Integer labels; region identity lives in pop_labels.npy, which readNPY cannot
% read (string dtype), and is reattached in plot_connectivity_matrix.py.
neuron_names = arrayfun(@(i) num2str(i), 0:n_neurons-1, 'UniformOutput', false);

fprintf('Allen: %d units, %d spikes in the first %g s\n', ...
    n_neurons, numel(spkt), max_time_ms / 1000);

% Group spike times by neuron
spike_times_cell = cell(1, n_neurons);
for i = 0:n_neurons-1
    spike_times_cell{i+1} = sort(spkt(spkid == i))';
end

% Build zero-padded spike time matrix (rows = spike#, cols = channel)
max_spikes = max(cellfun(@numel, spike_times_cell));
spike_matrix = zeros(max_spikes, n_neurons);
for i = 1:n_neurons
    st = spike_times_cell{i};
    spike_matrix(1:numel(st), i) = st;
end

fprintf('Spike matrix size: %d x %d  (rows=spike#, cols=channel)\n', ...
    size(spike_matrix, 1), size(spike_matrix, 2));

% Verify with raster plot (92 units: no per-neuron tick labels)
figure;
hold on;
for i = 1:n_neurons
    st = spike_times_cell{i};
    plot(st, i * ones(size(st)), 'k|', 'MarkerSize', 2);
end
xlabel('Time (ms)'); ylabel('Neuron index');
title('Allen hippocampal spontaneous — Spike Raster (1 ms precision)');
set(gca, 'YDir', 'reverse');

% Run CTM-DI
% Output conn_3d: (n_neurons x n_neurons x n_bin_widths)
conn_3d = CTM_spiketime_wrapper(spike_matrix);

fprintf('Connectivity matrix size: %d x %d x %d\n', size(conn_3d));

neuron_names_str = string(neuron_names);

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
save([save_dir '/ctm_di_raw_connectivity.mat'], 'conn_3d', 'neuron_names_str', '-v7.3');
fprintf('Saved %s/ctm_di_raw_connectivity.mat\n', save_dir);

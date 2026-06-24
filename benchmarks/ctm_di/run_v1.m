addpath('C:/Users/agupta17/Documents/GitHub/npy-matlab/npy-matlab');
% Run CTM-DI on the V1 multi-area-model network.
% Mirrors load_v1_network() in src/circuit_gnn/data.py:
%   - spikes.npy is (S, 2) = [neuron_idx (0-based), spike_time_ms]
%   - neuron count taken from the ground-truth weight matrix
%   - spike times floored to integer ms (native 1 ms, downsample_factor 1)
%   - population label per neuron resolved from neuron_info.json + neuron_gids

data_dir = 'C:/Users/agupta17/Documents/GitHub/circuitGNN/datasets/v1';

% Neuron count from the ground-truth weights (keeps silent neurons aligned)
weights   = readNPY([data_dir '/ground_truth_weights.npy']);   % (N x N)
n_neurons = size(weights, 1);

% Spike events
spike_data = readNPY([data_dir '/spikes.npy']);   % (S x 2)
spike_idx  = double(spike_data(:, 1));            % neuron index (0-based)
spike_t    = floor(double(spike_data(:, 2)));     % spike time in ms

% Resolve population label per neuron index (mirrors the Python searchsorted)
gids = double(readNPY([data_dir '/neuron_gids.npy']));   % (N,) sorted gids
info = jsondecode(fileread([data_dir '/neuron_info.json']));
fns  = fieldnames(info);                                  % keys like 'x2', 'x3', ...
neuron_names = repmat({'unknown'}, 1, n_neurons);
for k = 1:numel(fns)
    gid = str2double(fns{k}(2:end));        % strip leading 'x' that jsondecode adds
    pos = find(gids == gid, 1);             % 1-based index == python idx + 1
    if ~isempty(pos) && pos <= n_neurons
        neuron_names{pos} = info.(fns{k}).pop;
    end
end

fprintf('V1 network: %d neurons, %d spikes, t_max = %.1f s\n', ...
    n_neurons, numel(spike_t), max(spike_t) / 1000);

% Group spike times by neuron
spike_times_cell = cell(1, n_neurons);
for i = 0:n_neurons-1
    spike_times_cell{i+1} = sort(spike_t(spike_idx == i))';
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

% Verify with raster plot (no per-neuron labels: too many neurons)
figure;
hold on;
for i = 1:n_neurons
    st = spike_times_cell{i};
    plot(st, i * ones(size(st)), 'k.', 'MarkerSize', 2);
end
xlabel('Time (ms)'); ylabel('Neuron Index');
title('V1 Network — Spike Raster (1 ms precision)');
set(gca, 'YDir', 'reverse');

% Run CTM-DI
% Output conn_3d: (n_neurons x n_neurons x n_bin_widths)
conn_3d = CTM_spiketime_wrapper(spike_matrix);

fprintf('Connectivity matrix size: %d x %d x %d\n', size(conn_3d));

neuron_names_str = string(neuron_names);

out_dir = [data_dir '/init'];
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
save([out_dir '/ctm_di_raw_connectivity.mat'], 'conn_3d', 'neuron_names_str', '-v7.3');
fprintf('Saved %s/ctm_di_raw_connectivity.mat\n', out_dir);

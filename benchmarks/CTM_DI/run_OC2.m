repo_dir = 'C:/Users/agupta17/Documents/GitHub/circuitGNN';
data_dir = [repo_dir '/datasets/OC_2'];
save_dir = [data_dir '/baseline'];

% Native resolution of the OC_2 voltage traces (ms per sample)
native_ms = 0.5;

% Spike detection, matched to load_OC_2_network in src/circuit_gnn/data.py
thr = -10;   % mV
ra  = 15;    % refractory period in samples

% Neuron names from the header file (V[B20....] -> B20)
head_lines = readlines([data_dir '/spikes.head']);
head_lines = head_lines(startsWith(head_lines, 'V['));
neuron_names = arrayfun(@(s) extractBefore(extractAfter(s, '['), '.'), ...
    head_lines, 'UniformOutput', false);
n_neurons = numel(neuron_names);

% Load voltage traces: column 1 is time, next n_neurons columns are neurons
V = readmatrix([data_dir '/spikes.out'], 'FileType', 'text', 'Delimiter', '\t');
V = V(:, 2:1+n_neurons);

fprintf('Voltage matrix size: %d samples x %d neurons\n', size(V, 1), size(V, 2));

% Threshold crossings with refractory period, converted to spike times in ms
spike_times_cell = cell(1, n_neurons);
for i = 1:n_neurons
    mv = V(:, i);
    sp = find(mv > thr);
    if ~isempty(sp)
        sp = sp([true; diff(sp) > ra]);
    end
    spike_times_cell{i} = sp * native_ms;   % sample index (1-based) -> ms
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
fprintf('Spikes per neuron: %s\n', mat2str(cellfun(@numel, spike_times_cell)));

% Verify with raster plot
figure;
hold on;
for i = 1:n_neurons
    st = spike_times_cell{i};
    plot(st, i * ones(size(st)), 'k|', 'MarkerSize', 4);
end
yticks(1:n_neurons); yticklabels(neuron_names);
xlabel('Time (ms)'); ylabel('Neuron');
title('OC\_2 Network — Spike Raster (0.5 ms precision)');
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

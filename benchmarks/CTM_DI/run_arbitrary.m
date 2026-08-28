repo_dir = 'C:/Users/agupta17/Documents/GitHub/circuitGNN';
data_dir = [repo_dir '/datasets/arbitrary'];
save_dir = [data_dir '/baseline'];

% Native resolution of the SNNAP voltage traces (ms per sample)
native_ms  = 1.0;
time_max_s = 300;   % load_arbitrary_network keeps only the first 300 s
n_neurons  = 11;

% Spike detection, matched to load_arbitrary_network in src/circuit_gnn/data.py
thr = -10;   % mV
ra  = 8;     % re-arm period in samples (~8 ms)

neuron_names = arrayfun(@(i) num2str(i), 0:n_neurons-1, 'UniformOutput', false);

% Load the SNNAP dump: column 1 is time in SECONDS, the next n_neurons columns
% are membrane voltages. Whitespace-separated, matching Python's line.split().
M = readmatrix([data_dir '/spikes.out'], 'FileType', 'text', ...
    'Delimiter', {' ', '\t'}, 'ConsecutiveDelimitersRule', 'join', ...
    'LeadingDelimitersRule', 'ignore');

% Python appends the first row at/after time_max_s and then breaks, so that row
% is included
last = find(M(:, 1) >= time_max_s, 1, 'first');
if isempty(last)
    last = size(M, 1);
end
V = M(1:last, 2:1+n_neurons);

fprintf('Voltage matrix size: %d samples x %d neurons\n', size(V, 1), size(V, 2));

% Threshold crossings with re-arm period, converted to spike times in ms.
% Sample indices stay 1-based: spiketime2train discards spikes at t == 0
% because 0 doubles as the zero-padding sentinel in spike_matrix.
spike_times_cell = cell(1, n_neurons);
for i = 1:n_neurons
    mv = V(:, i);
    sp = find(mv > thr);
    if ~isempty(sp)
        sp = sp([true; diff(sp) > ra]);
    end
    spike_times_cell{i} = sp * native_ms;
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
title('Arbitrary Network — Spike Raster (1 ms precision)');
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

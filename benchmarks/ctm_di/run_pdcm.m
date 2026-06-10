addpath('C:/Users/agupta17/Documents/GitHub/npy-matlab/npy-matlab');
data_dir = 'C:/Users/agupta17/Documents/GitHub/circuitGNN/datasets/pdcm';

% Load spike data
spkt   = readNPY([data_dir '/spkt.npy']);   % spike times in ms
spkid  = readNPY([data_dir '/spkid.npy']);  % neuron ID for each spike (0-indexed)

% Floor to 1 ms integer bin (equivalent to downsample_factor=1 in Python)
spkt = floor(spkt);

n_neurons    = double(max(spkid)) + 1;
neuron_names = arrayfun(@(i) num2str(i), 0:n_neurons-1, 'UniformOutput', false);

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

% Verify with raster plot
figure;
hold on;
for i = 1:n_neurons
    st = spike_times_cell{i};
    plot(st, i * ones(size(st)), 'k|', 'MarkerSize', 4);
end
yticks(1:n_neurons); yticklabels(neuron_names);
xlabel('Time (ms)'); ylabel('Neuron');
title('PDCM Network — Spike Raster (1 ms precision)');
set(gca, 'YDir', 'reverse');

% Run CTM-DI
% Output conn_3d: (n_neurons x n_neurons x n_bin_widths)
conn_3d = CTM_spiketime_wrapper(spike_matrix);

fprintf('Connectivity matrix size: %d x %d x %d\n', size(conn_3d));

neuron_names_str = string(neuron_names);

cd('C:/Users/agupta17/Documents/GitHub/circuitGNN/datasets/pdcm')
save('ctm_di_raw_connectivity.mat', 'conn_3d', 'neuron_names_str', '-v7.3');
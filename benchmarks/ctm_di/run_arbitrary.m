% Run CTM-DI on the "arbitrary" SNNAP network.
% Mirrors load_arbitrary_network() in src/circuit_gnn/data.py:
%   - 11 neurons, native resolution 1 ms, first 300 s only
%   - voltage > -10 mV with an 8-sample re-arm period -> spike
%   - spike times floored to integer ms (native 1 ms == downsample_factor 1)

data_dir  = 'C:/Users/agupta17/Documents/GitHub/circuitGNN/datasets/arbitrary';
data_loc  = [data_dir '/spikes.out'];
num_neurons = 11;
time_max  = 300;    % seconds
thr       = -10;    % mV threshold
ra        = 8;      % re-arm period in samples (at 1 ms native resolution)

% Load voltage data (col 1 = time in s, cols 2..end = neuron voltages).
% Take only the first num_neurons voltage columns (mirrors volts[:, 1:12]).
raw       = readmatrix(data_loc, 'FileType', 'text', 'Delimiter', '\t');
time_s    = raw(:, 1);
keep      = time_s <= time_max;            % limit to first 300 s
raw       = raw(keep, :);
voltage   = raw(:, 2:(num_neurons + 1))';  % (num_neurons x n_timepoints)
time_vec  = raw(:, 1) * 1000;              % seconds -> ms (1 ms steps)

neuron_names = arrayfun(@(i) num2str(i), 0:num_neurons-1, 'UniformOutput', false);

fprintf('Detected %d neurons (using %d), %d timepoints (<= %d s)\n', ...
    size(raw, 2) - 1, num_neurons, size(voltage, 2), time_max);

% Detect spikes: threshold + re-arm period, then floor to integer ms.
spike_times_cell = cell(1, num_neurons);
for i = 1:num_neurons
    mv = voltage(i, :);
    sp = find(mv > thr);
    sp = sp([true, diff(sp) > ra]);
    spike_times_cell{i} = floor(time_vec(sp))';
end

% Build zero-padded spike time matrix (rows = spike#, cols = channel)
max_spikes = max(cellfun(@numel, spike_times_cell));
spike_matrix = zeros(max_spikes, num_neurons);
for i = 1:num_neurons
    st = spike_times_cell{i};
    spike_matrix(1:numel(st), i) = st;
end

fprintf('Spike matrix size: %d x %d  (rows=spike#, cols=channel)\n', ...
    size(spike_matrix, 1), size(spike_matrix, 2));

% Verify with raster plot
figure;
hold on;
for i = 1:num_neurons
    st = spike_times_cell{i};
    plot(st, i * ones(size(st)), 'k|', 'MarkerSize', 4);
end
yticks(1:num_neurons); yticklabels(neuron_names);
xlabel('Time (ms)'); ylabel('Neuron');
title('Arbitrary Network — Spike Raster (1 ms precision)');
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

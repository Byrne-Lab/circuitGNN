% Run CTM-DI on the "small_circuit_A" SNNAP network (Cai et al.).
% Mirrors load_small_circuit_A_network() in src/circuit_gnn/data.py:
%   - 3 neurons, native resolution 0.5 ms
%   - voltage > -10 mV with a 15-sample refractory period -> spike
%   - spike times floored to integer ms (0.5 ms -> 1 ms, downsample_factor 2)

data_dir  = 'C:/Users/agupta17/Documents/GitHub/circuitGNN/datasets/small_circuit_A';
data_loc  = [data_dir '/spikes.out'];
thr       = -10;    % mV threshold
ra        = 15;     % refractory period in samples (at 0.5 ms native resolution)

% Read neuron names from spikes.head (lines like "V[a....]<{svr} ...")
fid = fopen([data_dir '/spikes.head'], 'r');
neuron_names = {};
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && startsWith(line, 'V[')
        tok = regexp(line, 'V\[(\w+)\.+\]', 'tokens');
        if ~isempty(tok), neuron_names{end+1} = tok{1}{1}; end
    end
end
fclose(fid);
n_neurons = numel(neuron_names);

% Load voltage data (col 1 = time in s, cols 2..n_neurons+1 = voltages)
raw      = readmatrix(data_loc, 'FileType', 'text', 'Delimiter', '\t');
voltage  = raw(:, 2:(n_neurons + 1))';   % (n_neurons x n_timepoints)
time_vec = raw(:, 1) * 1000;             % seconds -> ms (0.5 ms steps)

fprintf('Detected %d neurons in file\n', n_neurons);

% Detect spikes: threshold + refractory period, then floor to integer ms.
spike_times_cell = cell(1, n_neurons);
for i = 1:n_neurons
    mv = voltage(i, :);
    sp = find(mv > thr);
    sp = sp([true, diff(sp) > ra]);
    spike_times_cell{i} = floor(time_vec(sp))';
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
title('Small Circuit A — Spike Raster (1 ms precision)');
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

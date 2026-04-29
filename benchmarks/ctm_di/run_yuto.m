%% Yuto Network — CTM-DI Connectivity Analysis
% Raw resolution: 0.5 ms (0.0005 s) per sample

data_loc = 'C:/Users/agupta17/Documents/GitHub/circuitGNN/datasets/yuto/spikes.out';
time_max  = 240;       % seconds
bin_size  = 0.0005;    % 0.5 ms raw bin (seconds)
thr       = -10;       % mV threshold
ra        = 15;        % refractory period in samples

% Read neuron order from spikes.head
fid = fopen('C:/Users/agupta17/Documents/GitHub/circuitGNN/datasets/yuto/spikes.head', 'r');
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

% Load voltage data
raw      = readmatrix(data_loc, 'FileType', 'text', 'Delimiter', '\t');
voltage  = raw(:, 2:end)';   % drop time col; readmatrix strips trailing tab unlike pandas
time_vec = raw(:, 1) * 1000;   % first column, converted to ms

fprintf('Detected %d neurons in file\n', size(voltage, 1));

% Detect spikes: threshold + refractory period
spike_times_cell = cell(1, n_neurons);
for i = 1:n_neurons
    mv = voltage(i, :);
    sp = find(mv > thr);
    sp = sp([true, diff(sp) > ra]);
    spike_times_cell{i} = time_vec(sp)';
end

% Build zero-padded spike time matrix (rows = spike#, cols = channel)
max_spikes = max(cellfun(@numel, spike_times_cell));

spike_matrix = zeros(max_spikes, n_neurons);   % zero-padded

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
title('Yuto Network — Spike Raster');
set(gca, 'YDir', 'reverse');
xlim([0 time_max * 1000]);


%  Output conn_3d: (n_neurons x n_neurons x n_bin_widths)

conn_3d = CTM_spiketime_wrapper(spike_matrix);

fprintf('Connectivity matrix size: %d x %d x %d\n', size(conn_3d));

neuron_names_str = string(neuron_names);

cd('C:/Users/agupta17/Documents/GitHub/circuitGNN/datasets/yuto/init')
save('ctm_di_raw_connectivity.mat', 'conn_3d', 'neuron_names_str', '-v7.3');
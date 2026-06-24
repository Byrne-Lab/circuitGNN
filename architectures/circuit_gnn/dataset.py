import torch


class SpikeTrainDataset(torch.utils.data.Dataset):
    def __init__(self, spike_data, window_size):
        # spike_data: tensor of shape (num_neurons, num_time)
        # window_size: number of time steps per window
        self.spike_data = spike_data
        self.window_size = window_size
        self.num_neurons, self.time_points = spike_data.shape
        self.num_windows = self.time_points - window_size

    def __len__(self):
        return self.num_windows

    def __getitem__(self, index):
        start = index
        end = start + self.window_size
        input_window = self.spike_data[:, start:end]
        target = self.spike_data[:, end].clamp(max=1.0)
        return input_window, target


class MultiStepSpikeTrainDataset(torch.utils.data.Dataset):
    """Returns k future time steps as targets instead of just t+1.

    targets shape: (num_neurons, k_steps), clamped to {0, 1}.
    """

    def __init__(self, spike_data, window_size, k_steps=4):
        self.spike_data = spike_data
        self.window_size = window_size
        self.k_steps = k_steps
        self.num_neurons, self.time_points = spike_data.shape
        self.num_windows = max(0, self.time_points - window_size - k_steps + 1)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, index):
        start = index
        end = start + self.window_size
        input_window = self.spike_data[:, start:end]
        targets = self.spike_data[:, end:end + self.k_steps].clamp(max=1.0)  # (N, k_steps)
        return input_window, targets

import torch

# pytorch dataset for windows
class SpikeTrainDataset(torch.utils.data.Dataset):
    def __init__(self, spike_data, window_size): # called when dataset is created
        # spike_data: tensor of shape (num_channels, num_time)
        # window_size: number of time steps per window
        self.spike_data = spike_data
        self.window_size = window_size
        self.num_neurons, self.time_points = spike_data.shape
        self.num_windows = self.time_points - window_size
    
    def __len__(self): # returns the number of windows, len(dataset) = #
        return self.num_windows
    
    def __getitem__(self, index): # return index-th window, dataset[i] calls __getitem__, dataset[0] passes index = 0
        start = index
        end = start + self.window_size
        input_window = self.spike_data[:, start:end]
        target = self.spike_data[:, end]
        return input_window, target

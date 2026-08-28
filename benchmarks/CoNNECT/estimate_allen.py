import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # benchmarks/CoNNECT
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))     # repo root
sys.path.insert(0, os.path.join(HERE, "modules"))          # CoNNECT modules
sys.path.insert(0, os.path.join(REPO, "src"))              # circuit_gnn package

from circuit_gnn.data import load_allen_network

NATIVE_BIN_MS = 1.0    # native resolution of this dataset (ms)
DATA_DIR = "datasets/allen/prep_793224716"
MAX_TIME_S = 600       # same window as scripts/run_allen.py and run_spycon.py


def write_spiketrain(binned_spikes, native_bin_ms, path):
    """Write binned spikes to CoNNECT's ';'-separated format, spike times
    floored to integer 1 ms (highest resolution = 1 ms)."""
    with open(path, "w") as f:
        for n in range(binned_spikes.shape[0]):
            idx = np.nonzero(binned_spikes[n])[0]
            counts = binned_spikes[n][idx].astype(int)
            times = np.floor(np.repeat(idx, counts) * native_bin_ms).astype(int)
            for t in times:
                f.write(f"{t}\n")
            f.write(";\n")


def main():
    out_dir = os.path.join(HERE, "results", "allen")
    os.makedirs(out_dir, exist_ok=True)
    spikefile = os.path.join(out_dir, "spiketrain.txt")
    savefile = os.path.join(out_dir, "estimated.csv")

    os.chdir(REPO)  # loaders read datasets/... relative to the repo root
    binned_spikes, true_conn, neuron_names = load_allen_network(
        downsample_factor=1, data_dir=DATA_DIR, max_time_s=MAX_TIME_S)
    print(f"[allen] {binned_spikes.shape[0]} neurons, first {MAX_TIME_S} s, "
          "spike times floored to 1 ms")
    write_spiketrain(binned_spikes, NATIVE_BIN_MS, spikefile)

    # in-vivo data: true_conn is an all-zeros placeholder, nothing to score against
    with open(os.path.join(out_dir, "neuron_names.txt"), "w") as f:
        f.write("\n".join(str(x) for x in neuron_names))

    os.chdir(HERE)  # estimate() loads model/ and modules/ relative to the CoNNECT dir
    import estimate as connect
    connect.estimate(spikefile, savefile)
    print(f"[allen] estimated PSP matrix saved to {savefile}")


if __name__ == "__main__":
    main()

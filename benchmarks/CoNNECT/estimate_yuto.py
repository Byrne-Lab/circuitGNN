import os
import sys
import importlib
import numpy as np

DATASET        = "yuto"
LOADER_NAME    = "load_yuto_network"
NATIVE_BIN_MS  = 0.5     # native resolution of this dataset (ms)
MIN_S          = 2       # CoNNECT center-deletion window (ms)
CONN_THRESHOLD = 0.5     # confidence threshold for a connection
N_THRED        = 1       # parallel threads for cross-correlogram computation

HERE = os.path.dirname(os.path.abspath(__file__))          # benchmarks/CoNNECT
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))     # repo root
sys.path.insert(0, os.path.join(HERE, "modules"))          # CoNNECT modules
sys.path.insert(0, os.path.join(REPO, "src"))              # circuit_gnn package


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
    data_mod = importlib.import_module("circuit_gnn.data")
    loader = getattr(data_mod, LOADER_NAME)

    out_dir = os.path.join(HERE, "results", DATASET)
    os.makedirs(out_dir, exist_ok=True)
    spikefile = os.path.join(out_dir, "spiketrain.txt")
    savefile = os.path.join(out_dir, "estimated.csv")

    # loaders use paths relative to the repo root (datasets/...)
    os.chdir(REPO)
    binned_spikes, true_conn, neuron_names = loader(downsample_factor=1)

    print(f"[{DATASET}] {binned_spikes.shape[0]} neurons, spike times floored to 1 ms")
    write_spiketrain(binned_spikes, NATIVE_BIN_MS, spikefile)

    # save the ground truth alongside for convenience
    np.save(os.path.join(out_dir, "true_conn.npy"), true_conn)
    with open(os.path.join(out_dir, "neuron_names.txt"), "w") as f:
        f.write("\n".join(str(x) for x in neuron_names))

    # estimate() loads "model/model.h5" and "modules/" relative to the CoNNECT dir
    os.chdir(HERE)
    import estimate as connect
    connect.estimate(spikefile, savefile,
                     min_s=MIN_S, conn_threshold=CONN_THRESHOLD, N_thred=N_THRED)

    print(f"[{DATASET}] estimated PSP matrix saved to {savefile}")


if __name__ == "__main__":
    main()

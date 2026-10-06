"""Settings shared by the paper's scripts: paths and the dataset registry.

The consensus settings (ablation threshold 0.10, 80% sign consistency, 20 ms sign bin,
rounding to 3 decimals) are the circuit_gnn defaults.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "paper" / "figures"


@dataclass(frozen=True)
class Dataset:
    loader: str                   # loader function in paper.data
    native_ms: float              # width of one sample of the raw recording
    num_runs: int                 # replicates to train
    num_epochs: int
    downsample: Union[int, str] = "auto"   # training bin in native samples; 'auto' = optimal_bin_ms
    kwargs: dict = field(default_factory=dict)   # extra loader arguments
    kind: str = "small"           # heatmap style: 'small' (labelled) or 'pdcm' (population blocks)
    round_decimals: Optional[int] = 3


# Manuscript names: small_circuit_A/B = Networks A/B, arbitrary = Network C,
# arbitrary_modification1/3 = Network C weakened/strengthened, OC_1 = Aplysia feeding circuit.
DATASETS = {
    "small_circuit_A": Dataset("load_small_circuit_A_network", 0.5, 50, 20),
    "small_circuit_B": Dataset("load_small_circuit_B_network", 0.5, 50, 20),
    "arbitrary": Dataset("load_arbitrary_network", 1.0, 50, 20),
    "arbitrary_modification1": Dataset("load_arbitrary_modification1_network", 1.0, 50, 20),
    "arbitrary_modification3": Dataset("load_arbitrary_modification3_network", 1.0, 50, 20),
    "OC_1": Dataset("load_OC_1_network", 0.5, 50, 20),
    "pdcm": Dataset("load_pdcm_network", 1.0, 50, 50, kind="pdcm"),
    # hidden-neuron and size/duration sweeps: 20 replicates at the 3 ms bin chosen for pdcm
    **{f"pdcm_hide{pct}": Dataset("load_pdcm_network", 1.0, 20, 50, downsample=3, kind="pdcm",
                                  kwargs={"hide_frac": pct / 100, "hide_seed": 0, "plot": False})
       for pct in (20, 40)},
    **{f"pdcm_{size}_{dur}s": Dataset("load_pdcm_network", 1.0, 20, 50, downsample=3, kind="pdcm",
                                      kwargs={"size": size, "max_time_s": dur, "plot": False})
       for size in ("0p002", "0p010") for dur in (60, 120, 240, 480)},
}

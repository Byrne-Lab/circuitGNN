"""Changed and control edge strengths for Network C's two modifications.

Run after `python -m paper.evaluate` for arbitrary, arbitrary_modification1 and
arbitrary_modification3:

    python -m paper.experiments.arbitrary_modification

Each modification changes one edge; its matched control has the same target and an unchanged
true weight. Prints both edges in the baseline and modified networks, as ground truth
(SNNAP maximal conductances, µS) and as inferred (the consensus means of mean_consistent.npy).
"""
import numpy as np
import pandas as pd

from paper.config import RESULTS_DIR
from paper.data import ARBITRARY_EDGES, MODIFICATION1_EDGES, MODIFICATION3_EDGES

MODIFICATIONS = [
    ("arbitrary_modification1", MODIFICATION1_EDGES, (2, 9), (4, 9)),
    ("arbitrary_modification3", MODIFICATION3_EDGES, (5, 1), (8, 1)),
]


def edge_name(edge):
    return f"N{edge[0] + 1}->N{edge[1] + 1}"


def main():
    gnn_baseline = np.load(RESULTS_DIR / "arbitrary" / "mean_consistent.npy")

    rows = []
    for name, edges, changed, control in MODIFICATIONS:
        gnn_modified = np.load(RESULTS_DIR / name / "mean_consistent.npy")
        for role, edge in (("changed", changed), ("control", control)):
            rows.append({"modification": name, "edge": edge_name(edge), "role": role,
                         "true_baseline_uS": ARBITRARY_EDGES[edge],
                         "true_modified_uS": edges[edge],
                         "inferred_baseline": gnn_baseline[edge],
                         "inferred_modified": gnn_modified[edge]})

    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.5g}".format))


if __name__ == "__main__":
    main()

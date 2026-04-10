"""
Plot a learned GNN adjacency matrix (gnn_x.npy) in the same format
as the PDCM ground truth connectivity plot.

Usage:
    python scripts/plot_pdcm_gnn.py results/pdcm/no_init/gnn_1.npy
"""

import sys
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from circuit_gnn.utils import plot_pdcm_matrix

data_dir = 'datasets/pdcm'


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_pdcm_gnn.py <path/to/gnn_x.npy>")
        sys.exit(1)

    gnn_path = Path(sys.argv[1])
    learned_A = np.load(gnn_path)
    pop_labels = np.load(f'{data_dir}/pop_labels.npy')

    plot_pdcm_matrix(learned_A, pop_labels,
                     method_name=gnn_path.stem,
                     save_dir=str(gnn_path.parent))


if __name__ == '__main__':
    main()
# python scripts/plot_pdcm_gnn.py results/pdcm/no_init/gnn_4.npy

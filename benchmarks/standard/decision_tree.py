# Pozo-Jimenez et al. (2021) decision-tree connectivity method.
# Python approximation of C5.0 (R package) using sklearn DecisionTreeClassifier.
#
# Key correspondences to the paper's R code (Afferents_learning repo):
#   categorize.R     → prepare_intervals  (A-E subinterval binary features)
#   thanos_snap.R    → _undersample       (1:4 negative undersampling)
#   call_c50.R       → run_analysis       (entropy criterion, cost matrix)
#   C5imp(usage)     → _feature_usage     (split-node count per feature)
#   do_analysis.R    → run_analysis       (preprocess→model→predict→evaluate)
#
# Feature encoding: paper uses one nominal categorical per neuron {0,A,B,C,D,E};
# we use 5 binary features per neuron (one-hot of A-E), which is equivalent for
# a decision tree (multi-category splits decompose into binary splits).
# Features are clipped to binary: the paper's categorize.R records WHICH bin
# a spike fell in, not how many spikes — so 0/1 per subinterval is correct.

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score

_SUB_LABELS = ['A', 'B', 'C', 'D', 'E']


def prepare_intervals(binned_spikes, target_idx, window_bins=50, subinterval_bins=10):
    """
    Build a tabular dataset from spike trains for predicting target neuron firing.

    Positive intervals: 50-bin window immediately before each target spike.
    Negative intervals: non-overlapping 50-bin chunks in the remaining time.
    Features: binary (0/1) per source neuron per subinterval (A=closest, E=farthest).
    Clipped to binary to match paper's categorize.R: the label records WHICH bin
    the spike fell in, not a count.

    Returns
    -------
    X             : ndarray (n_intervals, n_sources * n_sub)
    y             : ndarray (n_intervals,)   1=positive, 0=negative
    feature_names : list[str]  e.g. ['U1_A', 'U1_B', ..., 'U3_E', ...]
    source_indices: list[int]
    """
    num_channels, num_time = binned_spikes.shape
    n_sub = window_bins // subinterval_bins
    source_indices = [i for i in range(num_channels) if i != target_idx]

    target = binned_spikes[target_idx]
    positive_times = np.where(target == 1)[0]
    positive_times = positive_times[positive_times >= window_bins]

    # Mark bins consumed by positive intervals so negatives don't overlap
    used = np.zeros(num_time, dtype=bool)
    for t in positive_times:
        used[t - window_bins: t + 1] = True

    def _feats(t_end):
        row = []
        for src in source_indices:
            for s in range(n_sub):
                # s=0 → A (0–10 bins before target), s=n_sub-1 → E (farthest)
                end = t_end - s * subinterval_bins
                start = end - subinterval_bins
                if start < 0:
                    row.append(0)
                else:
                    # Binary: did any spike occur in this subinterval? (not count)
                    row.append(1 if np.any(binned_spikes[src, start:end]) else 0)
        return row

    rows, labels = [], []
    for t in positive_times:
        rows.append(_feats(t))
        labels.append(1)

    # Stride through unused bins to build negative intervals
    t = window_bins
    while t < num_time:
        if not used[t]:
            rows.append(_feats(t))
            labels.append(0)
            t += window_bins
        else:
            t += 1

    X = np.array(rows, dtype=np.float64)
    y = np.array(labels, dtype=int)
    feature_names = [
        f'U{src + 1}_{_SUB_LABELS[s]}'
        for src in source_indices
        for s in range(n_sub)
    ]
    return X, y, feature_names, source_indices


def _feature_usage(clf, n_features):
    """
    Usage-based variable importance: fraction of internal (non-leaf) split nodes
    that use each feature.  Matches C5.0's C5imp(model, metric='usage') which
    counts how many rules/conditions in the tree reference each variable.
    Returns a normalized array in [0, 1].
    """
    node_features = clf.tree_.feature   # -2 for leaf nodes
    counts = np.zeros(n_features)
    for f in node_features:
        if f >= 0:
            counts[f] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def _undersample(X, y, ratio, rng):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_neg = min(len(neg_idx), ratio * len(pos_idx))
    neg_sel = rng.choice(neg_idx, size=n_neg, replace=False)
    idx = rng.permutation(np.concatenate([pos_idx, neg_sel]))
    return X[idx], y[idx]


def run_analysis(X, y, feature_names, n_runs=30, undersample_ratio=4,
                 fn_cost=3.5, fp_cost=1.0, train_frac=0.8, seed=0):
    """
    Run C5.0-style analysis n_runs times with random undersampling.

    Follows paper protocol: snap data (1:ratio), 80/20 train/val split,
    cost-sensitive class weights, evaluated on the complete dataset.

    Returns
    -------
    dict with mean_mcc, sem_mcc, mean_precision, mean_recall,
    and variable_importance {feature_name: mean_importance}.
    """
    n_features = X.shape[1]
    if np.sum(y == 1) < 2 or np.sum(y == 0) < 2:
        return {
            'mean_mcc': 0.0, 'sem_mcc': 0.0,
            'mean_precision': 0.0, 'mean_recall': 0.0,
            'variable_importance': {fn: 0.0 for fn in feature_names},
        }

    mccs, precisions, recalls = [], [], []
    importances = np.zeros((n_runs, n_features))

    for run in range(n_runs):
        rng = np.random.default_rng(seed + run)
        X_snap, y_snap = _undersample(X, y, undersample_ratio, rng)
        n_train = max(2, int(len(y_snap) * train_frac))
        X_tr, y_tr = X_snap[:n_train], y_snap[:n_train]

        if len(np.unique(y_tr)) < 2:
            continue

        clf = DecisionTreeClassifier(
            criterion='entropy',
            class_weight={0: fp_cost, 1: fn_cost},
            random_state=int(rng.integers(1_000_000)),
        )
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X)
        mccs.append(matthews_corrcoef(y, y_pred))
        precisions.append(precision_score(y, y_pred, zero_division=0))
        recalls.append(recall_score(y, y_pred, zero_division=0))
        importances[run] = _feature_usage(clf, n_features)

    if not mccs:
        return {
            'mean_mcc': 0.0, 'sem_mcc': 0.0,
            'mean_precision': 0.0, 'mean_recall': 0.0,
            'variable_importance': {fn: 0.0 for fn in feature_names},
        }

    return {
        'mean_mcc': float(np.mean(mccs)),
        'sem_mcc': float(np.std(mccs) / np.sqrt(len(mccs))),
        'mean_precision': float(np.mean(precisions)),
        'mean_recall': float(np.mean(recalls)),
        'variable_importance': dict(zip(feature_names, importances.mean(axis=0).tolist())),
    }


def analyze_target(binned_spikes, target_idx, window_bins=50, subinterval_bins=10,
                   n_runs=30, undersample_ratio=4, fn_cost=3.5, fp_cost=1.0):
    """
    Full pipeline for one target neuron — mirrors do_analysis.R:
      preprocess (prepare_intervals) → model+predict+evaluate (run_analysis)

    Returns the run_analysis result dict plus the raw X, y, feature_names,
    and source_indices so callers can do further analysis (iterative, combinatory).
    """
    X, y, feature_names, source_indices = prepare_intervals(
        binned_spikes, target_idx,
        window_bins=window_bins, subinterval_bins=subinterval_bins,
    )
    result = run_analysis(
        X, y, feature_names,
        n_runs=n_runs, undersample_ratio=undersample_ratio,
        fn_cost=fn_cost, fp_cost=fp_cost,
    )
    result['X'] = X
    result['y'] = y
    result['feature_names'] = feature_names
    result['source_indices'] = source_indices
    return result


def calculate_mcc_matrix(binned_spikes, window_bins=50, subinterval_bins=10, n_runs=30):
    """
    Individual MCC matrix: mcc_matrix[i, j] = MCC of source i predicting target j.

    Matches the paper's "individual metrics" workflow: one model per source-target
    pair trained using only that source's subinterval features.
    """
    print('Calculating decision tree MCC matrix (individual)...')
    num_channels = binned_spikes.shape[0]
    mcc_matrix = np.zeros((num_channels, num_channels))
    n_sub = window_bins // subinterval_bins

    for target_idx in range(num_channels):
        if target_idx % 2 == 0:
            print(f'  target {target_idx}/{num_channels}')

        X_full, y, feature_names, source_indices = prepare_intervals(
            binned_spikes, target_idx,
            window_bins=window_bins, subinterval_bins=subinterval_bins,
        )
        if np.sum(y == 1) < 5:
            continue

        for k, src_idx in enumerate(source_indices):
            cols = slice(k * n_sub, (k + 1) * n_sub)
            X_src = X_full[:, cols]
            if X_src.max() == 0:
                continue
            fnames = feature_names[k * n_sub:(k + 1) * n_sub]
            result = run_analysis(X_src, y, fnames, n_runs=n_runs)
            mcc_matrix[src_idx, target_idx] = max(0.0, result['mean_mcc'])

    np.fill_diagonal(mcc_matrix, 0)
    return mcc_matrix


def calculate_variable_importance_matrix(binned_spikes, window_bins=50, subinterval_bins=10, n_runs=30):
    """
    Variable importance matrix: vi_matrix[i, j] = total importance of source i
    when a full model (all sources as features) predicts target j.

    Matches the paper's "direct run" workflow; per-source importance is the sum
    of usage scores (_feature_usage, analogous to C5imp) across that source's
    subinterval features.
    """
    print('Calculating decision tree variable importance matrix (full model)...')
    num_channels = binned_spikes.shape[0]
    vi_matrix = np.zeros((num_channels, num_channels))
    n_sub = window_bins // subinterval_bins

    for target_idx in range(num_channels):
        if target_idx % 2 == 0:
            print(f'  target {target_idx}/{num_channels}')

        X, y, feature_names, source_indices = prepare_intervals(
            binned_spikes, target_idx,
            window_bins=window_bins, subinterval_bins=subinterval_bins,
        )
        if np.sum(y == 1) < 5:
            continue

        result = run_analysis(X, y, feature_names, n_runs=n_runs)
        vi = result['variable_importance']

        for src_idx in source_indices:
            total = sum(
                vi.get(f'U{src_idx + 1}_{_SUB_LABELS[s]}', 0.0)
                for s in range(n_sub)
            )
            vi_matrix[src_idx, target_idx] = total

    np.fill_diagonal(vi_matrix, 0)
    return vi_matrix

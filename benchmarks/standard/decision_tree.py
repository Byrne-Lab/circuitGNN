import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score

_SUB_LABELS = ['A', 'B', 'C', 'D', 'E']


def prepare_intervals(binned_spikes, target_idx, window_bins=50, subinterval_bins=10):
    num_channels, num_time = binned_spikes.shape
    n_sub = window_bins // subinterval_bins # 50 //10 = 5
    source_indices = [i for i in range(num_channels) if i != target_idx] # every neuron except the target

    positive_times = np.where(binned_spikes[target_idx] == 1)[0] # every bin where target fires
    positive_times = positive_times[positive_times >= window_bins] # start 50th bin onwards

    used = np.zeros(num_time, dtype=bool)
    for t in positive_times: # for every target spike
        used[t - window_bins: t + 1] = True # reserve its 50 bin history

    def _feats(t_end): # for a given interval
        row = []
        for src in source_indices: # for all neurons
            for s in range(n_sub): # for each subinterval
                end   = t_end - s * subinterval_bins
                start = end - subinterval_bins
                row.append(0 if start < 0 else int(np.any(binned_spikes[src, start:end])))
        return row

    rows, labels = [], []
    for t in positive_times:
        rows.append(_feats(t))
        labels.append(1)

    t = window_bins
    while t < num_time:
        if not used[t]:
            rows.append(_feats(t))
            labels.append(0)
            t += window_bins
        else:
            t += 1

    X = np.array(rows, dtype=np.float64) # (n_intervals, (n-1)*5)
    y = np.array(labels, dtype=int)
    feature_names = [f'U{src + 1}_{_SUB_LABELS[s]}' for src in source_indices for s in range(n_sub)]
    return X, y, feature_names, source_indices


def _undersample(X, y, ratio, rng):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    neg_sel = rng.choice(neg_idx, size=min(len(neg_idx), ratio * len(pos_idx)), replace=False)
    idx = rng.permutation(np.concatenate([pos_idx, neg_sel]))
    return X[idx], y[idx]


def run_analysis(X, y, feature_names, n_runs=30, undersample_ratio=4, fn_cost=3.5, fp_cost=1.0, train_frac=0.8, seed=1):
    n_features = X.shape[1]
    empty = {'mean_mcc': 0.0, 'sem_mcc': 0.0, 'mean_precision': 0.0, 'mean_recall': 0.0, 'variable_importance': {fn: 0.0 for fn in feature_names}}

    mccs, precisions, recalls = [], [], []
    importances = np.zeros((n_runs, n_features))

    for run in range(n_runs):
        rng = np.random.default_rng(seed + run)
        X_snap, y_snap = _undersample(X, y, undersample_ratio, rng)
        n_train = int(len(y_snap) * train_frac)
        X_tr, y_tr = X_snap[:n_train], y_snap[:n_train]

        clf = DecisionTreeClassifier(criterion='entropy',
                                     class_weight={0: fp_cost, 1: fn_cost},
                                     random_state=int(rng.integers(1_000_000)))
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X)

        mccs.append(matthews_corrcoef(y, y_pred))
        precisions.append(precision_score(y, y_pred, zero_division=0))
        recalls.append(recall_score(y, y_pred, zero_division=0))
        importances[run] = clf.feature_importances_

    if not mccs:
        return empty

    return {
        'mean_mcc':          float(np.mean(mccs)),
        'sem_mcc':           float(np.std(mccs) / np.sqrt(len(mccs))),
        'mean_precision':    float(np.mean(precisions)),
        'mean_recall':       float(np.mean(recalls)),
        'variable_importance': dict(zip(feature_names, importances.mean(axis=0).tolist())),
    }


def analyze_target(binned_spikes, target_idx, window_bins=50, subinterval_bins=10,
                   n_runs=30, undersample_ratio=4, fn_cost=3.5, fp_cost=1.0):
    
    X, y, feature_names, source_indices = prepare_intervals(binned_spikes, target_idx, window_bins=window_bins, subinterval_bins=subinterval_bins)

    result = run_analysis(X, y, feature_names, n_runs=n_runs, undersample_ratio=undersample_ratio, fn_cost=fn_cost, fp_cost=fp_cost)
    result.update({'X': X, 'y': y, 'feature_names': feature_names, 'source_indices': source_indices})
    
    return result


def calculate_mcc_matrix(binned_spikes, window_bins=50, subinterval_bins=10, n_runs=30):
    print('Calculating decision tree MCC matrix...')
    num_channels = binned_spikes.shape[0]
    mcc_matrix = np.zeros((num_channels, num_channels))
    n_sub = window_bins // subinterval_bins

    for target_idx in range(num_channels):
        if target_idx % 5 == 0:
            print(f'  target {target_idx}/{num_channels}')

        X_full, y, feature_names, source_indices = prepare_intervals(binned_spikes, target_idx, window_bins=window_bins, subinterval_bins=subinterval_bins)
        if np.sum(y == 1) < 5:
            continue
        for k, src_idx in enumerate(source_indices):
            X_src = X_full[:, k * n_sub:(k + 1) * n_sub]
            if X_src.max() == 0:
                continue
            result = run_analysis(X_src, y, feature_names[k * n_sub:(k + 1) * n_sub], n_runs=n_runs)
            mcc_matrix[src_idx, target_idx] = max(0.0, result['mean_mcc'])

    np.fill_diagonal(mcc_matrix, 0)
    return mcc_matrix


def calculate_variable_importance_matrix(binned_spikes, window_bins=50, subinterval_bins=10, n_runs=30):
    print('Calculating decision tree variable importance matrix...')
    num_channels = binned_spikes.shape[0]
    vi_matrix = np.zeros((num_channels, num_channels))
    n_sub = window_bins // subinterval_bins

    for target_idx in range(num_channels):
        if target_idx % 2 == 0:
            print(f'  target {target_idx}/{num_channels}')
        X, y, feature_names, source_indices = prepare_intervals(binned_spikes, target_idx, window_bins=window_bins, subinterval_bins=subinterval_bins)
        if np.sum(y == 1) < 5:
            continue

        vi = run_analysis(X, y, feature_names, n_runs=n_runs)['variable_importance']
        for src_idx in source_indices:
            vi_matrix[src_idx, target_idx] = sum(vi.get(f'U{src_idx + 1}_{_SUB_LABELS[s]}', 0.0) for s in range(n_sub))

    np.fill_diagonal(vi_matrix, 0)
    return vi_matrix
TARGET_OCC = 0.015          # desired mean spikes per (neuron, bin)
OCC_BAND = (0.012, 0.020)   # preferred occupancy range


def occupancy_per_bin(spikes_native, factor):
    """Mean spikes per (neuron, bin) after rebinning native spikes by ``factor``."""
    n, T = spikes_native.shape
    factor = max(1, int(factor))
    trim = T // factor * factor
    if trim == 0:
        return float(spikes_native.mean())
    return float(spikes_native[:, :trim].reshape(n, -1, factor).sum(-1).mean())


def optimal_downsample_factor(spikes_native, native_ms, target_occupancy=TARGET_OCC, occ_band=OCC_BAND, max_bin_ms=20.0, verbose=True):
    """Integer downsample factor whose occupancy is closest to ``target_occupancy``.

    Factors landing inside ``occ_band`` are preferred; ties break toward the
    coarser bin (over-fine bins are the phantom-edge failure mode). Returns
    ``(factor, occupancy, info)`` where ``info`` holds the per-factor rows.
    """
    lo, hi = occ_band
    max_factor = max(1, round(max_bin_ms / native_ms))
    rows = [(f, f * native_ms, occupancy_per_bin(spikes_native, f))
            for f in range(1, max_factor + 1)]
    in_band = [r for r in rows if lo <= r[2] <= hi]
    factor, bin_ms, occ = min(in_band or rows,
                              key=lambda r: (abs(r[2] - target_occupancy), -r[0]))

    info = {"rows": rows, "bin_ms": bin_ms, "used_band": bool(in_band)}
    if verbose:
        print(f"[auto bin] target occ={target_occupancy} band={occ_band} "
              f"-> factor={factor} (bin={bin_ms:g} ms, occ={occ:.4f})")
        if not in_band:
            print(f"[auto bin] WARNING: no factor in band; used closest (occ={occ:.4f}).")
    return factor, occ, info


def resolve_downsample(downsample_factor, spikes_native, native_ms, **kwargs):
    """Coerce a loader's ``downsample_factor`` to an int; resolves ``'auto'``."""
    if isinstance(downsample_factor, str):
        if downsample_factor.lower() != "auto":
            raise ValueError(f"downsample_factor must be int or 'auto', got {downsample_factor!r}")
        return optimal_downsample_factor(spikes_native, native_ms, **kwargs)[0]
    return int(downsample_factor)


# native bin width (ms) per dataset loader, for the standalone recommender
_NATIVE_MS = {"arbitrary": 1.0, "yuto": 0.5, "pdcm": 1.0, "v1": 1.0,
              "small_circuit_A": 0.5, "small_circuit_B": 0.5}


def recommend_for_dataset(name, **kwargs):
    """Load ``name`` at native resolution and report its occupancy-optimal bin."""
    from circuit_gnn import data as _data  # lazy to avoid circular import
    if name not in _NATIVE_MS:
        raise KeyError(f"unknown dataset {name!r}; options: {list(_NATIVE_MS)}")
    spikes_native, _, _ = getattr(_data, f"load_{name}_network")(downsample_factor=1)
    return optimal_downsample_factor(spikes_native, _NATIVE_MS[name], **kwargs)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    for name in ["arbitrary", "yuto", "pdcm"]:
        print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
        factor, occ, info = recommend_for_dataset(name)
        print(f"  factor={factor}  bin={info['bin_ms']:g} ms  occ={occ:.4f}")

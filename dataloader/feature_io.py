"""Resolve paths and load arrays for 1s segment .npz features (baseline + ActionFormer)."""
import glob
import os
import warnings
from pathlib import Path

import numpy as np


def find_segment_feature_npz(segment_features_directory: str, backbone_subdir: str, recording_id: str) -> str:
    """
    Prefer CaptainCook canonical name: {recording_id}_360p.mp4_1s_1s.npz
    Fallback: glob if filenames differ (e.g. *_360p_224.mp4).
    """
    feat_root = Path(segment_features_directory) / "features" / backbone_subdir
    canonical = feat_root / f"{recording_id}_360p.mp4_1s_1s.npz"
    if canonical.is_file():
        return str(canonical)
    patterns = [
        str(feat_root / f"{recording_id}*_1s_1s.npz"),
        str(feat_root / f"{recording_id}*.npz"),
    ]
    matches = []
    for p in patterns:
        matches.extend(glob.glob(p))
    matches = sorted(set(matches))
    # Require "{recording_id}_" prefix so e.g. glob "1_10*" cannot pick "1_100_*.npz".
    prefix = f"{recording_id}_"
    filtered = [m for m in matches if Path(m).name.startswith(prefix)]
    if filtered:
        matches = filtered
    if not matches:
        raise FileNotFoundError(
            f"No feature .npz for recording_id={recording_id!r} under {feat_root}. "
            f"Tried {canonical.name} and globs {patterns}"
        )
    if len(matches) > 1:
        warnings.warn(
            f"Multiple .npz matches for recording_id={recording_id!r} under {feat_root}; "
            f"using {matches[0]}. Prefer canonical name {canonical.name} or remove duplicates.",
            stacklevel=2,
        )
    return matches[0]


def find_segment_npz_in_directory(
    feat_folder: str,
    recording_id: str,
    file_prefix: str = "",
    file_ext: str = ".npz",
) -> str:
    """
    Resolve a segment feature file inside a flat directory (ActionFormer error dataset).

    Tries CaptainCook canonical ``{prefix}{id}_360p.mp4_1s_1s.npz``, then globs for alternate
    stems (e.g. ``*_360p_224.mp4`` from PerceptionEncoder_feature_extractor).
    """
    feat_root = Path(feat_folder).resolve()
    ext = file_ext if file_ext.startswith(".") else f".{file_ext}"
    pf = file_prefix or ""
    canonical = feat_root / f"{pf}{recording_id}_360p.mp4_1s_1s{ext}"
    if canonical.is_file():
        return str(canonical)
    patterns = [
        str(feat_root / f"{pf}{recording_id}*_1s_1s{ext}"),
        str(feat_root / f"{pf}{recording_id}*{ext}"),
    ]
    matches = []
    for p in patterns:
        matches.extend(glob.glob(p))
    matches = sorted(set(matches))
    basename_prefix = f"{pf}{recording_id}_"
    filtered = [m for m in matches if Path(m).name.startswith(basename_prefix)]
    if filtered:
        matches = filtered
    if not matches:
        raise FileNotFoundError(
            f"No feature {ext} for recording_id={recording_id!r} under {feat_root}. "
            f"Tried {canonical.name} and globs {patterns}"
        )
    if len(matches) > 1:
        warnings.warn(
            f"Multiple feature matches for recording_id={recording_id!r} under {feat_root}; "
            f"using {matches[0]}. Prefer canonical {canonical.name}.",
            stacklevel=2,
        )
    return matches[0]


def load_segment_features_from_npz(npz_path: str) -> np.ndarray:
    """
    Load (T, D) float32 features from .npz produced by extractors or Omnivore pipeline.

    Key order: features (PerceptionEncoder extractor), feats (ActionFormer dataset), arr_0,
    then first 2D array skipping 'timestamps'.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        if "features" in data.files:
            arr = data["features"]
        elif "feats" in data.files:
            arr = data["feats"]
        elif "arr_0" in data.files:
            arr = data["arr_0"]
        else:
            arr = None
            for k in data.files:
                if k == "timestamps":
                    continue
                candidate = data[k]
                if hasattr(candidate, "ndim") and candidate.ndim == 2:
                    arr = candidate
                    break
            if arr is None:
                k0 = data.files[0]
                arr = data[k0]

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D features in {npz_path}, got shape {arr.shape}")
    return arr

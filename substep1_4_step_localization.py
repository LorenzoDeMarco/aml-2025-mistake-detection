import argparse
import glob
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def find_feature_file(features_dir: Path, recording_id: str) -> Path:
    """
    Try to locate the .npz feature file for a given recording_id.
    Typical patterns:
        {recording_id}*_1s_1s.npz
        {recording_id}*.npz
    """
    patterns = [
        str(features_dir / f"{recording_id}*_1s_1s.npz"),
        str(features_dir / f"{recording_id}*.npz"),
    ]
    matches = []
    for p in patterns:
        matches.extend(glob.glob(p))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(
            f"Cannot find feature file for recording_id='{recording_id}'.\n"
            f"features_dir: {features_dir}\n"
            f"Tried patterns: {patterns}"
        )
    return Path(matches[0])


def load_npz_array(npz_path: Path) -> np.ndarray:
    """
    Load (T, D) features from .npz.
    If saved by np.savez(path, array), default key is usually 'arr_0'.
    """
    d = np.load(npz_path, allow_pickle=False)
    if "feats" in d:
        arr = d["feats"]
    elif "arr_0" in d:
        arr = d["arr_0"]
    else:
        k0 = list(d.keys())[0]
        arr = d[k0]

    arr = np.asarray(arr)
    # Some pipelines may produce (T, 1, D); flatten it.
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    # If (D, T) accidentally, convert to (T, D) by heuristic
    # (Only apply when it looks like feature-dim first)
    if arr.ndim == 2 and arr.shape[0] <= 4096 and arr.shape[1] > arr.shape[0] * 2:
        arr = arr.T

    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)

    return arr.astype(np.float32)


def step_to_embedding(video_feats: np.ndarray, start_t: float, end_t: float, segment_sec: float) -> np.ndarray:
    """
    Convert a step time interval [start_t, end_t] into a single embedding by:
      1) mapping to segment indices
      2) slicing the corresponding segment features
      3) mean pooling across segments
    """
    start_idx = int(np.floor(start_t / segment_sec))
    end_idx = int(np.ceil(end_t / segment_sec))

    start_idx = max(0, min(start_idx, video_feats.shape[0] - 1))
    end_idx = max(start_idx + 1, min(end_idx, video_feats.shape[0]))

    seg = video_feats[start_idx:end_idx]  # (K, D)
    return seg.mean(axis=0)               # (D,)
# ================================================


def parse_eval_results(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    required = ["video-id", "t-start", "t-end", "label", "score"]
    for k in required:
        if k not in obj:
            raise KeyError(f"eval_results.pkl missing key '{k}'. got keys={list(obj.keys())}")

    n = len(obj["video-id"])
    for k in required[1:]:
        if len(obj[k]) != n:
            raise ValueError(f"Length mismatch: len(video-id)={n}, len({k})={len(obj[k])}")

    rows_by_video = defaultdict(list)
    for vid, s, e, lab, sc in zip(obj["video-id"], obj["t-start"], obj["t-end"], obj["label"], obj["score"]):
        vid = str(vid)
        rows_by_video[vid].append((float(s), float(e), int(lab), float(sc)))
    return rows_by_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_pkl", required=True, help="Path to eval_results.pkl")
    ap.add_argument("--features_dir", required=True, help="Directory containing per-video .npz feature files")
    ap.add_argument("--out_dir", required=True, help="Output directory for step-level embeddings")
    ap.add_argument("--segment_sec", type=float, default=1.0, help="Seconds per feature segment (default 1.0)")
    ap.add_argument("--score_thr", type=float, default=0.0, help="Filter predictions with score < score_thr")
    ap.add_argument("--topk", type=int, default=0, help="Keep top-K per video after score filtering (0 = keep all)")
    ap.add_argument("--min_len_sec", type=float, default=0.0, help="Drop segments shorter than this (seconds)")
    args = ap.parse_args()

    pkl_path = Path(args.eval_pkl)
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_video = parse_eval_results(pkl_path)

    num_videos = 0
    num_steps_total = 0

    for vid, rows in rows_by_video.items():
        # score filter + min length filter
        rows = [r for r in rows if r[3] >= args.score_thr and (r[1] - r[0]) >= args.min_len_sec]
        if not rows:
            continue
        # top-k by score (optional)
        if args.topk > 0 and len(rows) > args.topk:
            rows = sorted(rows, key=lambda x: x[3], reverse=True)[: args.topk]

        # sort by start time
        rows = sorted(rows, key=lambda x: x[0])
        # load features
        try:
            feat_path = find_feature_file(features_dir, vid)
        except FileNotFoundError as e:
            print(f"[WARN] {vid}: {e}")
            continue

        feats = load_npz_array(feat_path)  # (T, D)
        T, D = feats.shape

        segments = []
        labels = []
        scores = []
        embeddings = []

        for s, e, lab, sc in rows:
            s = max(0.0, float(s))
            e = max(s + 1e-4, float(e))
            emb = step_to_embedding(feats, s, e, args.segment_sec)  # (D,)

            segments.append([s, e])
            labels.append(lab)
            scores.append(sc)
            embeddings.append(emb)

        segments = np.asarray(segments, dtype=np.float32)           # (N,2)
        labels = np.asarray(labels, dtype=np.int64)                 # (N,)
        scores = np.asarray(scores, dtype=np.float32)               # (N,)
        embeddings = np.stack(embeddings, axis=0).astype(np.float32)  # (N,D)

        np.savez(out_dir / f"{vid}.npz",
                 video_id=vid,
                 segments=segments,
                 labels=labels,
                 scores=scores,
                 embeddings=embeddings,
                 feature_file=str(feat_path),
                 segment_sec=np.float32(args.segment_sec))

        num_videos += 1
        num_steps_total += segments.shape[0]
        print(f"[OK] {vid}: steps={segments.shape[0]} emb={embeddings.shape} feats(T,D)=({T},{D})")

    print(f"\nDone. videos={num_videos}, total_steps={num_steps_total}, out_dir={out_dir}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dataloader.feature_io import (
    find_segment_npz_in_directory,
    load_segment_features_from_npz,
)

# CaptainCook combined GT: same step_id appears at most this many times in one video.
MAX_LABEL_INSTANCES_PER_VIDEO = 4
# Same default as ActionFormer test_cfg.iou_threshold (libs/core/config.py).
TEMPORAL_NMS_IOU = 0.1

PredictionRow = Tuple[float, float, int, float]


def parse_eval_results(pkl_path: Path):
    """Load ActionFormer eval_results.pkl and group predictions by video id."""
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)

    required = ("video-id", "t-start", "t-end", "label", "score")
    missing = [k for k in required if k not in obj]
    if missing:
        raise KeyError(f"{pkl_path} missing keys {missing}; got keys={list(obj.keys())}")

    video_ids = [str(v) for v in obj["video-id"]]
    starts = np.asarray(obj["t-start"], dtype=np.float32).reshape(-1)
    ends = np.asarray(obj["t-end"], dtype=np.float32).reshape(-1)
    labels = np.asarray(obj["label"], dtype=np.int64).reshape(-1)
    scores = np.asarray(obj["score"], dtype=np.float32).reshape(-1)

    n = len(video_ids)
    if not (len(starts) == len(ends) == len(labels) == len(scores) == n):
        raise ValueError(
            "Inconsistent eval result lengths: "
            f"video-id={n}, t-start={len(starts)}, t-end={len(ends)}, "
            f"label={len(labels)}, score={len(scores)}"
        )

    rows_by_video = defaultdict(list)
    for vid, start, end, label, score in zip(video_ids, starts, ends, labels, scores):
        rows_by_video[vid].append((float(start), float(end), int(label), float(score)))
    return rows_by_video


def find_feature_file(features_dir: Path, recording_id: str, file_ext: str, file_prefix: str):
    return Path(
        find_segment_npz_in_directory(
            str(features_dir),
            recording_id,
            file_prefix=file_prefix,
            file_ext=file_ext,
        )
    )


def load_timestamps(npz_path: Path, num_features: int, segment_sec: float):
    """Return segment timestamps as (T, 2) seconds."""
    with np.load(npz_path, allow_pickle=False) as data:
        if "timestamps" in data.files:
            timestamps = np.asarray(data["timestamps"], dtype=np.float32)
            if timestamps.ndim == 2 and timestamps.shape[1] == 2:
                if timestamps.shape[0] == num_features:
                    return timestamps

        if "uniform_times" in data.files:
            centers = np.asarray(data["uniform_times"], dtype=np.float32).reshape(-1)
            if len(centers) == num_features:
                if len(centers) > 1:
                    delta = float(np.median(np.diff(centers)))
                else:
                    delta = float(segment_sec)
                return np.stack((centers - 0.5 * delta, centers + 0.5 * delta), axis=1)

    starts = np.arange(num_features, dtype=np.float32) * float(segment_sec)
    ends = starts + float(segment_sec)
    return np.stack((starts, ends), axis=1)


def average_features_for_segment(features, timestamps, start_sec: float, end_sec: float):
    """
    Average PE features that fall inside a predicted step.

    First uses fully-contained 1s feature segments. If none are fully contained,
    falls back to any overlap. If prediction is shorter than the feature stride,
    uses the nearest feature center.
    """
    seg_starts = timestamps[:, 0]
    seg_ends = timestamps[:, 1]

    inside = (seg_starts >= start_sec) & (seg_ends <= end_sec)
    selected = features[inside]
    count = int(inside.sum())

    if count == 0:
        overlap = (seg_ends > start_sec) & (seg_starts < end_sec)
        selected = features[overlap]
        count = int(overlap.sum())

    if count == 0:
        centers = 0.5 * (seg_starts + seg_ends)
        mid = 0.5 * (start_sec + end_sec)
        nearest = int(np.argmin(np.abs(centers - mid)))
        selected = features[nearest:nearest + 1]
        count = 1

    return selected.mean(axis=0).astype(np.float32), count


def load_step_description_mapping(step_json_path):
    if not step_json_path:
        return {}

    path = Path(step_json_path)
    if not path.exists():
        print(f"[WARN] step description json not found: {path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return {int(k): str(v) for k, v in obj.items()}


def slugify_recipe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def load_recipe_node_counts(avg_csv: Path, task_graph_dir: Path) -> Dict[str, int]:
    """Map recipe_id -> number of nodes in the CaptainCook task graph."""
    counts: Dict[str, int] = {}
    with avg_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            recipe_id = row[0].strip()
            if recipe_id.lower() in {"average", "avg"} or not recipe_id.isdigit():
                continue
            graph_path = task_graph_dir / f"{slugify_recipe_name(row[1].strip())}.json"
            if not graph_path.exists():
                raise FileNotFoundError(
                    f"Task graph not found for recipe_id={recipe_id}: {graph_path}"
                )
            with graph_path.open("r", encoding="utf-8") as graph_file:
                graph = json.load(graph_file)
            counts[recipe_id] = len(graph["steps"])
    if not counts:
        raise RuntimeError(f"No recipe node counts loaded from {avg_csv}")
    return counts


def recipe_id_from_video_id(video_id: str) -> str:
    return video_id.split("_")[0]


def segment_iou(seg_a: PredictionRow, seg_b: PredictionRow) -> float:
    start = max(seg_a[0], seg_b[0])
    end = min(seg_a[1], seg_b[1])
    inter = max(0.0, end - start)
    union = (seg_a[1] - seg_a[0]) + (seg_b[1] - seg_b[0]) - inter
    return inter / union if union > 0.0 else 0.0


def remove_exact_duplicate_boundaries(rows: List[PredictionRow]) -> Tuple[List[PredictionRow], int]:
    """Keep only the highest-score prediction for each exact (start, end) boundary."""
    best_by_boundary = {}
    for row in rows:
        key = (row[0], row[1])
        if key not in best_by_boundary or row[3] > best_by_boundary[key][3]:
            best_by_boundary[key] = row
    kept = list(best_by_boundary.values())
    return kept, len(rows) - len(kept)


def temporal_nms_per_label(rows: List[PredictionRow], iou_threshold: float) -> Tuple[List[PredictionRow], int]:
    """Within each label, suppress lower-score segments that overlap a kept segment."""
    by_label: Dict[int, List[PredictionRow]] = defaultdict(list)
    for row in rows:
        by_label[row[2]].append(row)

    kept: List[PredictionRow] = []
    for group in by_label.values():
        group = sorted(group, key=lambda x: x[3], reverse=True)
        selected: List[PredictionRow] = []
        for candidate in group:
            if all(segment_iou(candidate, kept_seg) <= iou_threshold for kept_seg in selected):
                selected.append(candidate)
        kept.extend(selected)
    return kept, len(rows) - len(kept)


def cap_per_label_instances(
    rows: List[PredictionRow],
    max_per_label: int,
) -> Tuple[List[PredictionRow], int]:
    """Keep at most max_per_label segments per label, ranked by score."""
    by_label: Dict[int, List[PredictionRow]] = defaultdict(list)
    for row in rows:
        by_label[row[2]].append(row)

    kept: List[PredictionRow] = []
    for group in by_label.values():
        group = sorted(group, key=lambda x: x[3], reverse=True)[:max_per_label]
        kept.extend(group)
    return kept, len(rows) - len(kept)


def cap_video_steps(rows: List[PredictionRow], max_steps: int) -> Tuple[List[PredictionRow], int]:
    """Keep at most max_steps segments for the whole video, ranked by score."""
    if max_steps <= 0 or len(rows) <= max_steps:
        return rows, 0
    kept = sorted(rows, key=lambda x: x[3], reverse=True)[:max_steps]
    return kept, len(rows) - len(kept)


def canonicalize_predictions(
    rows: List[PredictionRow],
    recipe_id: str,
    recipe_node_counts: Dict[str, int],
    score_thr: float,
    min_len_sec: float,
    topk_override: int = 0,
) -> Tuple[List[PredictionRow], Dict[str, int]]:
    """
    Normalize ActionFormer predictions before step embedding export.

    Policy (derived from CaptainCook GT + task graphs, not ad-hoc top-k):
      1. score / duration filter
      2. exact-boundary dedup
      3. per-label temporal NMS
      4. per-label instance cap (GT max repeat = 4)
      5. per-video cap = task-graph node count (or --topk override)
    """
    stats = {
        "filtered_score_or_len": 0,
        "removed_exact_boundary_dup": 0,
        "removed_temporal_nms": 0,
        "removed_per_label_cap": 0,
        "removed_video_cap": 0,
    }

    filtered = [
        (max(0.0, s), max(max(0.0, s) + 1e-4, e), lab, sc)
        for s, e, lab, sc in rows
        if sc >= score_thr and (e - s) >= min_len_sec
    ]
    stats["filtered_score_or_len"] = len(rows) - len(filtered)

    filtered, removed = remove_exact_duplicate_boundaries(filtered)
    stats["removed_exact_boundary_dup"] = removed

    filtered, removed = temporal_nms_per_label(filtered, TEMPORAL_NMS_IOU)
    stats["removed_temporal_nms"] = removed

    filtered, removed = cap_per_label_instances(filtered, MAX_LABEL_INSTANCES_PER_VIDEO)
    stats["removed_per_label_cap"] = removed

    if topk_override > 0:
        video_cap = topk_override
    else:
        video_cap = recipe_node_counts.get(recipe_id, max(recipe_node_counts.values()))

    filtered, removed = cap_video_steps(filtered, video_cap)
    stats["removed_video_cap"] = removed
    stats["video_cap"] = video_cap

    return sorted(filtered, key=lambda x: x[0]), stats


def main():
    ap = argparse.ArgumentParser(
        description="Convert ActionFormer predictions into step boundaries and step-level PE embeddings."
    )
    ap.add_argument("--eval_pkl", required=True, help="Path to eval_results.pkl or eval_results_filtered.pkl")
    ap.add_argument("--features_dir", required=True, help="Directory containing per-video PE .npz files")
    ap.add_argument("--out_dir", required=True, help="Output directory for per-video step-level .npz files")
    ap.add_argument("--segment_sec", type=float, default=1.0, help="Fallback seconds per feature segment")
    ap.add_argument("--score_thr", type=float, default=0.01, help="Drop predictions with score below this value")
    ap.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Optional hard cap on steps per video; 0 uses task-graph node count (recommended)",
    )
    ap.add_argument("--min_len_sec", type=float, default=0.0, help="Drop predictions shorter than this many seconds")
    ap.add_argument("--task_graph_dir", default="./captaincook/task_graphs", help="CaptainCook task graph JSON directory")
    ap.add_argument("--avg_csv", default="./captaincook/metadata/average_segment_length.csv", help="Recipe id/name CSV")
    ap.add_argument("--file_ext", default=".npz", help="Feature file extension")
    ap.add_argument("--file_prefix", default="", help="Optional feature filename prefix")
    ap.add_argument("--step_desc_json", default="", help="Optional label-id to step-description JSON")
    args = ap.parse_args()

    eval_pkl = Path(args.eval_pkl)
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_video = parse_eval_results(eval_pkl)
    step_desc_map = load_step_description_mapping(args.step_desc_json)
    recipe_node_counts = load_recipe_node_counts(Path(args.avg_csv), Path(args.task_graph_dir))

    boundaries_by_video = {}
    saved = 0
    skipped_no_predictions = 0
    skipped_missing_features = 0
    total_steps = 0
    aggregate_stats = defaultdict(int)

    for video_id, rows in sorted(rows_by_video.items()):
        rows, canon_stats = canonicalize_predictions(
            rows,
            recipe_id=recipe_id_from_video_id(video_id),
            recipe_node_counts=recipe_node_counts,
            score_thr=args.score_thr,
            min_len_sec=args.min_len_sec,
            topk_override=args.topk,
        )
        for key, value in canon_stats.items():
            aggregate_stats[key] += value
        if not rows:
            skipped_no_predictions += 1
            continue

        try:
            feat_path = find_feature_file(
                features_dir,
                video_id,
                file_ext=args.file_ext,
                file_prefix=args.file_prefix,
            )
        except FileNotFoundError as exc:
            print(f"[WARN] {video_id}: {exc}")
            skipped_missing_features += 1
            continue

        features = load_segment_features_from_npz(str(feat_path))
        timestamps = load_timestamps(feat_path, features.shape[0], args.segment_sec)

        segments = []
        labels = []
        scores = []
        embeddings = []
        feature_counts = []
        descriptions = []
        boundary_rows = []

        for start, end, label, score in rows:
            embedding, count = average_features_for_segment(features, timestamps, start, end)
            segments.append([start, end])
            labels.append(label)
            scores.append(score)
            embeddings.append(embedding)
            feature_counts.append(count)
            desc = step_desc_map.get(label, "")
            descriptions.append(desc)
            boundary_rows.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "label": int(label),
                    "score": float(score),
                    "feature_count": int(count),
                    "description": desc,
                }
            )

        segments = np.asarray(segments, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float32)
        embeddings = np.stack(embeddings, axis=0).astype(np.float32)
        feature_counts = np.asarray(feature_counts, dtype=np.int64)
        descriptions = np.asarray(descriptions, dtype=str)

        np.savez(
            out_dir / f"{video_id}.npz",
            video_id=np.asarray(video_id),
            segments=segments,
            labels=labels,
            scores=scores,
            embeddings=embeddings,
            feature_counts=feature_counts,
            descriptions=descriptions,
            feature_file=np.asarray(str(feat_path)),
            step_features=embeddings,
            step_starts=segments[:, 0],
            step_ends=segments[:, 1],
            step_labels=labels,
        )

        boundaries_by_video[video_id] = boundary_rows
        saved += 1
        total_steps += len(rows)
        print(
            f"[OK] {video_id}: steps={len(rows)} cap={canon_stats['video_cap']} "
            f"embeddings={embeddings.shape} feature_file={feat_path.name}"
        )

    with (out_dir / "step_boundaries.json").open("w", encoding="utf-8") as f:
        json.dump(boundaries_by_video, f, indent=2)

    print("\nDone.")
    print(f"saved_videos={saved}")
    print(f"total_steps={total_steps}")
    print(f"avg_steps_per_video={total_steps / saved:.2f}" if saved else "avg_steps_per_video=0")
    print(
        "canonicalization_removed="
        f"score_or_len={aggregate_stats['filtered_score_or_len']}, "
        f"exact_boundary={aggregate_stats['removed_exact_boundary_dup']}, "
        f"temporal_nms={aggregate_stats['removed_temporal_nms']}, "
        f"per_label_cap={aggregate_stats['removed_per_label_cap']}, "
        f"video_cap={aggregate_stats['removed_video_cap']}"
    )
    print(f"skipped_no_predictions={skipped_no_predictions}")
    print(f"skipped_missing_features={skipped_missing_features}")
    print(f"out_dir={out_dir}")
    print(f"boundaries_json={out_dir / 'step_boundaries.json'}")


if __name__ == "__main__":
    main()

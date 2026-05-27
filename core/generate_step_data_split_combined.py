"""
Generate annotations/data_splits/step_data_split_combined.json using the same
75% / 16% / 9% per-recording step split as CaptainCookStepDataset._init_step_split.

Requires er_annotations/recordings_combined_splits.json (see generate_recordings_combined_splits.py).
"""
import argparse
import json
import os
import sys

SPLIT_PROPORTION = (0.75, 0.16, 0.9)

RECORDINGS_SPLIT_PATH = "er_annotations/recordings_combined_splits.json"
STEP_ANN_PATH = "annotations/annotation_json/step_annotations.json"
OUTPUT_PATH = "annotations/data_splits/step_data_split_combined.json"


def _valid_steps(step_anns, recording_id):
    steps = []
    for step in step_anns[recording_id]["steps"]:
        if step["start_time"] < 0 or step["end_time"] < 0:
            continue
        steps.append(step)
    return steps


def _unique_step_ids_by_error(steps):
    """One entry per step_id, in annotation order (matches _prepare_recording_step_dictionary)."""
    step_has_errors = {}
    for step in steps:
        step_id = step["step_id"]
        if step_id not in step_has_errors:
            step_has_errors[step_id] = step["has_errors"]
    normal_step_ids = [step_id for step_id, has_err in step_has_errors.items() if not has_err]
    error_step_ids = [step_id for step_id, has_err in step_has_errors.items() if has_err]
    return normal_step_ids, error_step_ids


def _split_recording_steps(steps):
    normal_step_ids, error_step_ids = _unique_step_ids_by_error(steps)

    # Matches _init_step_split: shuffle calls are no-ops (list is discarded).
    num_normal = len(normal_step_ids)
    num_error = len(error_step_ids)
    normal_bounds = (
        int(num_normal * SPLIT_PROPORTION[0]),
        int(num_normal * (SPLIT_PROPORTION[0] + SPLIT_PROPORTION[1])),
    )
    error_bounds = (
        int(num_error * SPLIT_PROPORTION[0]),
        int(num_error * (SPLIT_PROPORTION[0] + SPLIT_PROPORTION[1])),
    )

    train_ids = normal_step_ids[: normal_bounds[0]] + error_step_ids[: error_bounds[0]]
    val_ids = (
        normal_step_ids[normal_bounds[0] : normal_bounds[1]]
        + error_step_ids[error_bounds[0] : error_bounds[1]]
    )
    test_ids = normal_step_ids[normal_bounds[1] :] + error_step_ids[error_bounds[1] :]
    return train_ids, val_ids, test_ids


def build_step_data_split(recording_ids, step_anns):
    result = {"train": [], "val": [], "test": []}
    for recording_id in recording_ids:
        if recording_id not in step_anns:
            continue
        steps = _valid_steps(step_anns, recording_id)
        train_ids, val_ids, test_ids = _split_recording_steps(steps)
        result["train"].extend(f"{recording_id}_{step_id}" for step_id in train_ids)
        result["val"].extend(f"{recording_id}_{step_id}" for step_id in val_ids)
        result["test"].extend(f"{recording_id}_{step_id}" for step_id in test_ids)
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate step_data_split_combined.json")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (kept for CLI compatibility; split order matches _init_step_split)",
    )
    args = parser.parse_args()
    _ = args.seed

    if not os.path.isfile(RECORDINGS_SPLIT_PATH):
        print(
            f"Missing {RECORDINGS_SPLIT_PATH}. Run: python core/generate_recordings_combined_splits.py",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(RECORDINGS_SPLIT_PATH, "r") as f:
        recordings_split = json.load(f)
    with open(STEP_ANN_PATH, "r") as f:
        step_anns = json.load(f)

    recording_ids = (
        recordings_split["train"]
        + recordings_split["val"]
        + recordings_split["test"]
    )
    result = build_step_data_split(recording_ids, step_anns)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    counts = {k: len(result[k]) for k in result}
    print(f"Wrote {OUTPUT_PATH} ({counts}, total={sum(counts.values())})")


if __name__ == "__main__":
    main()

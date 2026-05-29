#!/usr/bin/env python3
"""Count correct vs erroneous recipes from step annotations JSON.

Usage:
    python scripts/count_recipes.py \
        --annotations-file annotations/annotation_json/step_annotations.json \
        [--show-samples N]

Outputs counts and optionally example ids.
"""
from pathlib import Path
import json
import argparse
from typing import List


def load_annotations(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def analyze(data: dict):
    total = 0
    correct = []
    erroneous = []
    unknown = []

    for vid, entry in data.items():
        total += 1
        if isinstance(entry, dict) and "steps" in entry and isinstance(entry["steps"], list):
            steps = entry["steps"]
            any_err = any((isinstance(s, dict) and s.get("has_errors", False)) for s in steps)
            if any_err:
                erroneous.append(vid)
            else:
                correct.append(vid)
        else:
            unknown.append(vid)

    return {
        "total": total,
        "correct_count": len(correct),
        "erroneous_count": len(erroneous),
        "unknown_count": len(unknown),
        "correct_ids": correct,
        "erroneous_ids": erroneous,
        "unknown_ids": unknown,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-file", required=False,
                    default="annotations/annotation_json/step_annotations.json",
                    help="Path to step_annotations.json")
    ap.add_argument("--show-samples", type=int, default=0,
                    help="Show up to N example ids for each category")
    args = ap.parse_args(argv)

    p = Path(args.annotations_file)
    if not p.exists():
        raise SystemExit(f"Annotations file not found: {p}")

    data = load_annotations(p)
    res = analyze(data)

    print(f"TOTAL recordings: {res['total']}")
    print(f"CORRECT (no step errors): {res['correct_count']}")
    print(f"ERRONEOUS (any step has has_errors==true): {res['erroneous_count']}")
    print(f"UNKNOWN / no steps present: {res['unknown_count']}")

    if args.show_samples and args.show_samples > 0:
        n = args.show_samples
        def show(title: str, ids: List[str]):
            print(f"\n{title} (showing up to {n}):")
            for vid in ids[:n]:
                print(vid)

        show("EXAMPLES - CORRECT", res["correct_ids"])
        show("EXAMPLES - ERRONEOUS", res["erroneous_ids"])
        show("EXAMPLES - UNKNOWN", res["unknown_ids"])


if __name__ == "__main__":
    main()
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch

# Hungarian
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -----------------------------
# Helpers: mapping recipe_id -> task_graph json
# -----------------------------
def slugify_recipe_name(name: str) -> str:
    """'Microwave Egg Sandwich' -> 'microwaveeggsandwich'."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def load_recipeid_to_graphpath(avg_csv: Path, task_graph_dir: Path) -> Dict[str, Path]:
    """
    avg_csv rows like:
      4,Ramen,39.6939...
    returns recipe_id(str)-> task_graph_path
    """
    mapping = {}
    with avg_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            rid = row[0].strip()
            if rid.lower() in {"average", "avg"}:
                continue
            if not rid.isdigit():
                continue
            name = row[1].strip()
            slug = slugify_recipe_name(name)
            p = task_graph_dir / f"{slug}.json"
            if not p.exists():
                raise FileNotFoundError(
                    f"Task graph not found for recipe_id={rid}, name={name}, slug={slug}: {p}"
                )
            mapping[rid] = p
    if not mapping:
        raise RuntimeError(f"No recipe_id mapping loaded from {avg_csv}")
    return mapping


def recipe_id_from_video_id(video_id: str) -> str:
    """video_id like '1_7' -> '1'"""
    return video_id.split("_")[0]


# -----------------------------
# Load substep1 npz (visual steps)
# -----------------------------
def load_substep1_npz(npz_path: Path) -> Dict[str, Any]:
    d = np.load(npz_path, allow_pickle=False)
    if "embeddings" not in d.files or "segments" not in d.files:
        raise ValueError(f"{npz_path} missing embeddings/segments, keys={d.files}")

    step_emb = np.asarray(d["embeddings"], dtype=np.float32)  # (S,D)
    segments = np.asarray(d["segments"], dtype=np.float32)    # (S,2)

    if "video_id" in d.files:
        vid = str(d["video_id"]).strip()
        vid = vid.strip("[]'\" ")
        if not vid:
            vid = npz_path.stem
    else:
        vid = npz_path.stem

    return {"video_id": vid, "step_emb": step_emb, "segments": segments}


# -----------------------------
# Load task graph json (ONLY your format: steps+edges)
# -----------------------------
def load_task_graph_steps_edges(graph_json: Path) -> Dict[str, Any]:
    """
    ONLY supports:
      {
        "steps": {"0": "START", "1": "...", ...},
        "edges": [[0, 1], [1, 2], ...]
      }

    Returns:
      node_texts: List[str] ordered by ascending original node_id
      node_ids:   List[int] original node ids in the same order
      edge_index: LongTensor (2,E) using re-indexed node indices [0..V-1]
    """
    with graph_json.open("r", encoding="utf-8") as f:
        g = json.load(f)

    if "steps" not in g or not isinstance(g["steps"], dict):
        raise ValueError(f"Invalid graph: missing/invalid 'steps' in {graph_json}")
    if "edges" not in g or not isinstance(g["edges"], list):
        raise ValueError(f"Invalid graph: missing/invalid 'edges' in {graph_json}")

    steps_dict = g["steps"]
    node_ids = sorted(int(k) for k in steps_dict.keys())
    if len(node_ids) == 0:
        raise ValueError(f"Empty steps in {graph_json}")

    # robust: steps_dict keys are strings in your files
    node_texts = [str(steps_dict[str(nid)]) for nid in node_ids]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    e_pairs: List[Tuple[int, int]] = []
    for e in g["edges"]:
        if not (isinstance(e, (list, tuple)) and len(e) >= 2):
            continue
        u, v = int(e[0]), int(e[1])
        if u not in id_to_idx or v not in id_to_idx:
            raise ValueError(f"Edge references unknown node id in {graph_json}: {u}->{v}")
        e_pairs.append((id_to_idx[u], id_to_idx[v]))

    if len(e_pairs) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(e_pairs, dtype=torch.long).t().contiguous()  # (2,E)

    return {"node_texts": node_texts, "node_ids": node_ids, "edge_index": edge_index}


# -----------------------------
# PE-Core text encoder (open_clip)
# -----------------------------
@torch.no_grad()
def encode_text_pe_core(
    model, tokenizer, texts: List[str], device: torch.device, batch_size: int = 64, normalize: bool = True
) -> torch.Tensor:
    """Returns (V,D) float32 on CPU."""
    model = model.to(device).eval()
    outs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        tokens = tokenizer(chunk).to(device)
        emb = model.encode_text(tokens).float()
        if normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        outs.append(emb.detach().cpu())
    return torch.cat(outs, dim=0)


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# -----------------------------
# Hungarian matching
# -----------------------------
def hungarian_one_to_one(
    node_emb: torch.Tensor,  # (V,D)
    step_emb: torch.Tensor,  # (S,D)
    sim_threshold: float = -1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      node_to_step: (V,) matched step idx or -1
      step_to_node: (S,) matched node idx or -1

    We maximize cosine similarity => minimize cost = 1 - sim.
    We do rectangular assignment and then drop matches below sim_threshold (if threshold > -1).
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for Hungarian matching. Install: pip install scipy")

    n = l2norm(node_emb)
    s = l2norm(step_emb)
    sim = (n @ s.t()).detach().cpu().numpy()  # (V,S)
    cost = 1.0 - sim

    r_ind, c_ind = linear_sum_assignment(cost)

    V, S = node_emb.size(0), step_emb.size(0)
    node_to_step = torch.full((V,), -1, dtype=torch.long)
    step_to_node = torch.full((S,), -1, dtype=torch.long)

    for i, j in zip(r_ind, c_ind):
        if sim_threshold > -1.0 and sim[i, j] < sim_threshold:
            continue
        node_to_step[i] = int(j)
        step_to_node[j] = int(i)

    return node_to_step, step_to_node


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--substep1_dir", required=True, help="Directory with Substep1 .npz (step embeddings)")
    ap.add_argument("--task_graph_dir", required=True, help="Directory task_graphs/*.json")
    ap.add_argument("--avg_csv", required=True, help="average_segment_length.csv (recipe_id -> recipe name)")
    ap.add_argument("--out_dir", required=True, help="Output directory (per-video .pt)")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # PE-Core model id
    ap.add_argument("--pe_core_model_id", default="hf-hub:timm/PE-Core-B-16")
    ap.add_argument("--text_batch_size", type=int, default=64)
    ap.add_argument("--normalize", action="store_true", help="L2 normalize embeddings for matching")

    # matching
    ap.add_argument("--sim_threshold", type=float, default=-1.0,
                    help="Drop matches with cosine sim < threshold. Use -1 to keep all assigned pairs.")
    ap.add_argument("--topk_steps", type=int, default=0, help="Optional truncate step sequence (0 keep all)")

    args = ap.parse_args()

    substep1_dir = Path(args.substep1_dir)
    task_graph_dir = Path(args.task_graph_dir)
    avg_csv = Path(args.avg_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # mapping recipe_id -> graph path
    rid2graph = load_recipeid_to_graphpath(avg_csv, task_graph_dir)

    # PE-Core encoder
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(args.pe_core_model_id)
    tokenizer = open_clip.get_tokenizer(args.pe_core_model_id)

    # in-memory recipe cache (fast, no disk write)
    # recipe_id -> {"node_texts","node_ids","edge_index","node_text_emb"}
    recipe_cache: Dict[str, Dict[str, Any]] = {}

    saved = 0
    skipped_missing_graph = 0
    skipped_bad = 0

    for npz_path in sorted(substep1_dir.glob("*.npz")):
        try:
            item = load_substep1_npz(npz_path)
            video_id = item["video_id"]
            step_emb_np = item["step_emb"]
            segments_np = item["segments"]
        except Exception as e:
            print(f"[SKIP bad npz] {npz_path.name}: {e}")
            skipped_bad += 1
            continue

        recipe_id = recipe_id_from_video_id(video_id)
        if recipe_id not in rid2graph:
            print(f"[SKIP no recipe_id mapping] video_id={video_id} recipe_id={recipe_id}")
            skipped_missing_graph += 1
            continue

        # --- recipe cache: load/encode once per recipe
        if recipe_id not in recipe_cache:
            graph_path = rid2graph[recipe_id]
            try:
                g = load_task_graph_steps_edges(graph_path)
            except Exception as e:
                print(f"[SKIP bad graph] {graph_path.name} for recipe_id={recipe_id}: {e}")
                skipped_bad += 1
                continue

            node_texts = g["node_texts"]
            node_ids = g["node_ids"]
            edge_index = g["edge_index"]

            node_text_emb = encode_text_pe_core(
                model, tokenizer, node_texts, device=device,
                batch_size=args.text_batch_size, normalize=args.normalize
            )  # (V,Dt) CPU

            recipe_cache[recipe_id] = {
                "graph_file": str(graph_path),
                "node_texts": node_texts,
                "node_ids": node_ids,
                "edge_index": edge_index,
                "node_text_emb": node_text_emb,
            }

        rc = recipe_cache[recipe_id]
        edge_index = rc["edge_index"]
        node_text_emb = rc["node_text_emb"]
        node_texts = rc["node_texts"]
        node_ids = rc["node_ids"]

        # --- video visual steps
        step_x = torch.from_numpy(step_emb_np).float()          # (S,Dv)
        segments = torch.from_numpy(segments_np).float()        # (S,2)

        if args.topk_steps and step_x.size(0) > args.topk_steps:
            step_x = step_x[:args.topk_steps]
            segments = segments[:args.topk_steps]

        # --- matching (per-video)
        node_emb_dev = node_text_emb.to(device)
        step_emb_dev = step_x.to(device)
        if args.normalize:
            node_emb_dev = l2norm(node_emb_dev)
            step_emb_dev = l2norm(step_emb_dev)

        try:
            node_to_step, step_to_node = hungarian_one_to_one(
                node_emb_dev, step_emb_dev,
                sim_threshold=args.sim_threshold
            )
        except Exception as e:
            print(f"[SKIP match fail] video_id={video_id}: {e}")
            skipped_bad += 1
            continue

        node_to_step = node_to_step.cpu()
        step_to_node = step_to_node.cpu()

        # -----------------------------
        # Per-video graph package output
        # -----------------------------
        pack = {
            # meta
            "video_id": video_id,
            "recipe_id": recipe_id,
            "graph_file": rc["graph_file"],

            # graph structure
            "edge_index": edge_index,      # (2,E), node index in [0..V-1]
            "node_ids": node_ids,          # original node ids aligned with x_text
            "node_texts": node_texts,      # text aligned with x_text

            # node & step features
            "x_text": node_text_emb,       # (V,Dt)  node text embeddings
            "step_x": step_x,              # (S,Dv)  visual step embeddings
            "segments": segments,          # (S,2)   start/end times

            # matching
            "node_to_step": node_to_step,  # (V,) matched step idx or -1
            "step_to_node": step_to_node,  # (S,) matched node idx or -1
        }

        torch.save(pack, out_dir / f"{video_id}.pt")
        saved += 1
        if saved % 50 == 0:
            print(f"[OK] saved {saved}")

    print("\n=== Substep3 done ===")
    print("saved:", saved)
    print("skipped_missing_graph:", skipped_missing_graph)
    print("skipped_bad:", skipped_bad)
    if not HAS_SCIPY:
        print("WARNING: scipy not installed -> Hungarian matching will fail. Install: pip install scipy")


if __name__ == "__main__":
    main()

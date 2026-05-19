import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

# --- EgoVLP repo path ---
EGOVLP_REPO = "EgoVLP"
sys.path.insert(0, EGOVLP_REPO)

from model.model import FrozenInTime
from transformers import AutoTokenizer

RECIPE_MAPPING = {
    '1': 'microwaveeggsandwich.json',    '2': 'dressedupmeatballs.json',
    '3': 'microwavemugpizza.json',        '4': 'ramen.json',
    '5': 'coffee.json',                   '7': 'breakfastburritos.json',
    '8': 'spicedhotchocolate.json',       '9': 'microwavefrenchtoast.json',
    '10': 'pinwheels.json',               '12': 'tomatomozzarellasalad.json',
    '13': 'buttercorncup.json',           '15': 'tomatochutney.json',
    '16': 'scrambledeggs.json',           '17': 'cucumberraita.json',
    '18': 'zoodles.json',                 '20': 'sautedmushrooms.json',
    '21': 'blenderbananapancakes.json',   '22': 'herbomeletwithfriedtomatoes.json',
    '23': 'broccolistirfry.json',         '25': 'panfriedtofu.json',
    '26': 'mugcake.json',                 '27': 'cheesepimiento.json',
    '28': 'spicytunaavocadowraps.json',   '29': 'capresebruschetta.json'
}

SKIP_TOKENS = {'START', 'END'}


def load_egovlp_text_encoder(checkpoint_path, device):
    """
    Loads EgoVLP text encoder from the official pretrained checkpoint.

    The checkpoint (egovlp.pth) stores a full training state under the key
    'state_dict'. The model config must match the pretraining setup exactly.
    Text output dim: 256 (projected from DistilBERT 768 via a linear head).
    Visual and text embeddings are in the same aligned 256-dim space.

    Checkpoint download:
      https://drive.google.com/file/d/1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7
      Place it at: pretrained/egovlp.pth
    """
    # EgoVLP pretraining config — must match checkpoint architecture exactly
    model_config = {
        "video_params": {
            "model": "SpaceTimeTransformer",
            "arch_config": "base_patch16_224",
            "num_frames": 16,
            "pretrained": False,   # ImageNet init not needed, we load full checkpoint
            "time_init": "zeros"
        },
        "text_params": {
            "model": "distilbert-base-uncased",
            "pretrained": True,
            "input": "text"
        },
        "projection_dim": 256,
        "load_checkpoint": None   
    }

    model = FrozenInTime(**model_config)

    # egovlp.pth stores the full training state — extract only model weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # fallback if already a plain state_dict

    # strict=False ignores video-branch keys we don't need for text encoding
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded — missing keys: {len(missing)}, unexpected: {len(unexpected)}")

    model.eval().to(device)

    # DistilBERT tokenizer — same one used during EgoVLP pretraining
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    return model, tokenizer


def encode_texts(texts, model, tokenizer, device, batch_size=64):
    """
    Encodes a list of strings via EgoVLP text encoder.

    Internally calls model.compute_text(), which runs:
      DistilBERT -> mean pooling over tokens -> linear projection -> 256-dim output
    Output is L2-normalized to match the visual feature space.

    Returns: np.array of shape [N, 256], dtype float32
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77   # same max length used during EgoVLP pretraining
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            emb = model.compute_text(tokens)          # [B, 256]
            emb = F.normalize(emb, p=2, dim=-1)       # L2-normalize to unit sphere

        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def encode_recipe_nodes(task_graph_dir, model, tokenizer, device):
    """
    Encodes task graph node descriptions for each recipe.

    Each task graph JSON has a 'steps' dict mapping node_id -> description string.
    START and END nodes are filtered out — they carry no semantic content
    and would pollute the similarity computation during Hungarian matching.

    Returns: dict {recipe_prefix (str) -> np.array [N_nodes, 256]}
    Shape is FIXED per recipe — all videos of the same recipe share the same
    text features, since the task graph is recipe-level, not video-level.
    """
    recipe_text_feats = {}

    for prefix, filename in RECIPE_MAPPING.items():
        path = os.path.join(task_graph_dir, filename)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping recipe {prefix}")
            continue

        with open(path, "r") as f:
            graph = json.load(f)

        steps = graph["steps"]  # {"0": "START", "1": "Add-...", ..., "N": "END"}

        # Sort by numeric node id, filter START/END anchors
        node_ids = sorted(steps.keys(), key=lambda x: int(x))
        filtered = [(nid, steps[nid]) for nid in node_ids
                    if steps[nid] not in SKIP_TOKENS]

        if not filtered:
            print(f"WARNING: recipe {prefix} has no valid nodes after filtering")
            continue

        node_ids_filtered, node_texts = zip(*filtered)
        node_texts = list(node_texts)

        embeddings = encode_texts(node_texts, model, tokenizer, device)  # [N, 256]
        recipe_text_feats[prefix] = embeddings

        print(f"Recipe {prefix:>3} ({filename:<35}): {len(node_texts):>2} nodes -> {embeddings.shape}")

    return recipe_text_feats


def build_npz(visual_npz_path, recipe_text_feats, output_path):
    """
    Builds the output NPZ assigning to each video_id the text features
    of its corresponding recipe.

    Output per video: np.array [N_nodes_recipe, 256]
    All videos of the same recipe share identical text features — this is
    intentional and correct, since the task graph is recipe-level.
    """
    visual_data = np.load(visual_npz_path)
    video_ids = visual_data.files

    output = {}
    missing_recipes = []

    for vid in video_ids:
        prefix = vid.split("_")[0]
        if prefix in recipe_text_feats:
            output[vid] = recipe_text_feats[prefix]
        else:
            missing_recipes.append(vid)
            print(f"WARNING: no recipe text features for video {vid} (prefix={prefix})")

    np.savez(output_path, **output)

    print(f"\nSaved: {output_path}")
    print(f"Videos written : {len(output)} / {len(video_ids)}")
    if missing_recipes:
        print(f"Videos skipped : {missing_recipes}")

    # --- Sanity checks ---
    sample_vid    = list(output.keys())[0]
    sample_prefix = sample_vid.split("_")[0]
    same_recipe   = [v for v in output if v.split("_")[0] == sample_prefix]

    all_identical = all(
        np.array_equal(output[sample_vid], output[v]) for v in same_recipe
    )
    print(f"\nSanity — recipe {sample_prefix}: {len(same_recipe)} videos, all identical = {all_identical}")

    norms = np.linalg.norm(output[sample_vid], axis=-1)
    print(f"Sanity — node L2 norms (expect ~1.0): min={norms.min():.3f}  max={norms.max():.3f}  mean={norms.mean():.3f}")


if __name__ == "__main__":
    CHECKPOINT = "pretrained/egovlp.pth"             # download from EgoVLP release
    GRAPH_DIR  = "annotations/task_graphs"            
    VISUAL_NPZ = "step_embeddings_dataset.npz"        
    OUTPUT_NPZ = "text_task_graphs_v2.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = load_egovlp_text_encoder(CHECKPOINT, device)
    recipe_feats     = encode_recipe_nodes(GRAPH_DIR, model, tokenizer, device)
    build_npz(VISUAL_NPZ, recipe_feats, OUTPUT_NPZ)
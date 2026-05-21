import argparse
from pathlib import Path
import open_clip
import torch
import torch.nn as nn
import numpy as np
import json
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────
#  Hungarian matching
# ─────────────────────────────────────────────
def hungarian_matching(visual_embs: torch.Tensor,
                        text_embs: torch.Tensor,
                        sim_threshold: float = -1.0):
    
    # Cosine similarity matrix  (N x M)
    v_norm = torch.nn.functional.normalize(visual_embs, p=2, dim=1)
    t_norm = torch.nn.functional.normalize(text_embs,   p=2, dim=1)
    sim_matrix = np.dot(v_norm, t_norm.T)         # (N, M) matrix of cosine similarities high value = good match
 
    # Hungarian minimises cost -> negate similarity
    # use the negative similarity matrix to find the optimal assignment of visual embeddings to text embeddings.
    # row_ind and col_ind are the indices of the matched pairs in the original matrices.
    row_ind, col_ind = linear_sum_assignment(-sim_matrix) 
    visual_indices, node_indices, similarities = [], [], []
    for r, c in zip(row_ind, col_ind):
        s = sim_matrix[r, c]
        if s >= sim_threshold:
            visual_indices.append(int(r))
            node_indices.append(int(c))
            similarities.append(float(s))
 
    return visual_indices, node_indices, similarities #return  the index of video,teext and the similariities value 



def load_graph_step_edges(task_graph):
    steps= task_graph['steps'] # steps is a list of [step_id, step_text]
    edges= task_graph['edges']  # edges is a list of [source_step_id, target_step_id]
    
    step_id= list(steps.keys()) #  list of step_id
    step_text =list(steps.values()) #  list of step_text
    
    return step_id, step_text, edges

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def recipe_from__id(recipe_id):
    return recipe_id.split("_")[0]

def create_recipeid_to_json_map(avg_csv, task_graph_dir): 
    import csv
    recipeid_to_json = {}
    with open(avg_csv, 'r') as f:
        reader = csv.DictReader(f)
        #colonne = reader.fieldnames
        #print(f"Columns in {avg_csv}: {colonne}")
        for row in reader:
            recipe_id = row['activity_id'] 
            recipe_name = row['activity_name']
            if "=" in row['activity_id'] or "Average" in row['activity_id'] or row['activity_id'] == "":
                # Skip the average row which is not a real recipe
                continue
            
            #print(f"Mapping recipe_id {recipe_id} to recipe name '{recipe_name}'...")
            slugname = recipe_name.lower().replace(' ', '')#json file are in lower letter and no space
            json_path = task_graph_dir / f"{slugname}.json"
            if json_path.exists():
                recipeid_to_json[recipe_id] = json_path
            else:
                print(f"Warning: JSON file not found for recipe_id {recipe_id} (expected at {json_path})")
    return recipeid_to_json


def main():
    ap = argparse.ArgumentParser()


    ap.add_argument("--substep1_dir", required=True, help="Directory with Substep1 .npz (step embeddings)", default='../output_KFold_1s_step_embedding')
    ap.add_argument("--task_graph_dir", required=True, help="Directory task_graphs/*.json", default='../annotations/task_graphs')
    ap.add_argument("--avg_csv", required=True, help="average_segment_length.csv (recipe_id -> recipe name)", default='../annotations/annotation_csv/average_segment_length.csv')
    ap.add_argument("--out_dir", required=True, help="Output directory (per-video .pt)", default='./output_3_3')


    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")



    ap.add_argument("--text_batch_size", type=int, default=64)
    ap.add_argument("--normalize", action="store_true", help="L2 normalize embeddings for matching")


    # matching
    ap.add_argument("--sim_threshold", type=float, default=-1.0,
                    help="Drop matches with cosine sim < threshold. Use -1 to keep all assigned pairs.")
    #ap.add_argument("--topk_steps", type=int, default=0, help="Optional truncate step sequence (0 keep all)")


    args = ap.parse_args()

     
    substep1_dir = Path(args.substep1_dir)
    task_graph_dir = Path(args.task_graph_dir)
    avg_csv = Path(args.avg_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    projection = None  # lazy init after we know D

    device = torch.device(args.device)

    # I must obtain a map from recipe_id to recipe name from the avg_csv 
    map_recipeid_to_json= create_recipeid_to_json_map(avg_csv, task_graph_dir) #(1 ->microwaveegg)

    #---Load PE model----
    model_name= "hf-hub:timm/PE-Core-B-16" # Optimal solution for training
    print(f"Loading model {model_name} on device {device}...")
    model, _, _ = open_clip.create_model_and_transforms(model_name)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)

    #---Load npz-----
    for  npz in substep1_dir.glob("*.npz"):
        recipe_id = npz.stem # from the file 1_7.npz -> 1_7
        recipe= recipe_from__id(recipe_id) # from 1_7 -> 1
        if recipe not in map_recipeid_to_json:
            print(f"Skipping {recipe_id} as no corresponding JSON found.")
            continue
        recipe_path = map_recipeid_to_json[recipe]
        print(f"Processing {recipe_id} with JSON {recipe_path}...")

        #---Load step embeddings---
        data = np.load(npz)
        step_embeddings = torch.from_numpy(data['step_embedding']).to(device)  # (num_steps, embedding_dim)
        step_interval= torch.from_numpy(data['segments'])  # (num_steps, 2) start/end frame index for each step
        step_label = torch.from_numpy(data['label'])  # (num_steps,) step label index for each step
        
        #---Load task graph JSON---
        task_graph = load_json(recipe_path)

        #---Encode task graph text---
        step_ids, step_texts, edges = load_graph_step_edges(task_graph)
        

        text_embeddings = []
        for i in range(0, len(step_texts), args.text_batch_size):
            batch_texts = step_texts[i:i+args.text_batch_size]
            tokens = tokenizer(batch_texts).to(device)
            with torch.no_grad():
                batch_embeddings = model.encode_text(tokens)
                if args.normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                text_embeddings.append(batch_embeddings.to(device))
        
        text_embeddings = torch.cat(text_embeddings, dim=0)  # (num_nodes, embedding_dim)

        #---Save combined data without matching (for debugging)---
        #out_path = out_dir / f"{recipe_id}.pt"
        #torch.save({
        #    'step_embeddings': step_embeddings.to(device),
        #   'text_embeddings': text_embeddings.to(device),
        #    'task_graph': task_graph
        #}, out_path)
        #print(f"Saved combined data to {out_path}")

                
        vis_idx, node_idx, sims = hungarian_matching(
            step_embeddings, 
            text_embeddings, 
            sim_threshold=args.sim_threshold
        )
        
        print(f"  Matched {len(vis_idx)} / {min(len(text_embeddings), step_embeddings.shape[0])} pairs")

       
        out_path = out_dir / f"{recipe_id}.pt"
        torch.save({
            'visual_embeddings':       step_embeddings.to(device),
            'text_embeddings':         text_embeddings.to(device),
            
            #result of matching
            'matched_visual_indices':  vis_idx, # index of the matched step embedding
            'matched_node_indices':    node_idx, # index of the matched node in the graph
            'match_similarities':      sims,    # cosine similarity value
            
            # Metadati del task graph
            'step_ids':       step_ids,
            'step_texts':     step_texts,
            'edges':          edges,
            'step_intervals': step_interval.to(device),
            'step_labels':    step_label.to(device),
            'task_graph':     task_graph,
        }, out_path)
        
        print(f"  Saved (Pure Matching) -> {out_path}")
    






if __name__ == "__main__":
    main()
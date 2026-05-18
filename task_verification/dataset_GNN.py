import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

# Official mapping between video prefix and CaptainCook4D task graph JSON files
RECIPE_MAPPING = {
    '1': 'microwaveeggsandwich.json', '2': 'dressedupmeatballs.json', '3': 'microwavemugpizza.json',
    '4': 'ramen.json', '5': 'coffee.json', '7': 'breakfastburritos.json',
    '8': 'spicedhotchocolate.json', '9': 'microwavefrenchtoast.json', '10': 'pinwheels.json',
    '12': 'tomatomozzarellasalad.json', '13': 'buttercorncup.json', '15': 'tomatochutney.json',
    '16': 'scrambledeggs.json', '17': 'cucumberraita.json', '18': 'zoodles.json',
    '20': 'sautedmushrooms.json', '21': 'blenderbananapancakes.json', '22': 'herbomeletwithfriedtomatoes.json',
    '23': 'broccolistirfry.json', '25': 'panfriedtofu.json', '26': 'mugcake.json',
    '27': 'cheesepimiento.json', '28': 'spicytunaavocadowraps.json', '29': 'capresebruschetta.json'
}

class TaskVerificationGraphDataset(Dataset):
    def __init__(self, visual_npz_path, text_npz_path, graph_zip_path, annotations_path, video_ids, split='train'):
        """
        Graph-Aware Dataset optimized for Multi-Modal Procedural Mistake Detection.
        Uses on-the-fly np.load calls to remain process-safe during multi-worker execution.
        """
        self.split = split
        self.visual_npz_path = visual_npz_path
        self.text_npz_path = text_npz_path
        
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
            
        with np.load(visual_npz_path) as vis_data:
            vis_keys = set(k.replace('.npy', '') for k in vis_data.files)
            
        with np.load(text_npz_path) as text_data:
            text_keys = set(k.replace('.npy', '') for k in text_data.files)
            
        self.video_list = []
        for vid in video_ids:
            if vid in vis_keys and vid in text_keys and vid in self.annotations:
                self.video_list.append(vid)
                
        print(f"[{split.upper()}] Dataset initialized via np.load. Total active samples: {len(self.video_list)}")
                
        self.recipe_edges = {}
        base_graph_dir = graph_zip_path if os.path.isdir(graph_zip_path) else 'task_graphs'
        
        for prefix, filename in RECIPE_MAPPING.items():
            file_path = os.path.join(base_graph_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.recipe_edges[prefix] = data.get('edges', [])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_id = self.video_list[idx]
        recipe_prefix = video_id.split('_')[0]
        
        with np.load(self.visual_npz_path) as vis_data:
            vis_feat = vis_data[video_id].astype(np.float32)
            
        #.npy extension
        with np.load(self.text_npz_path) as text_data:
            text_key = video_id if video_id in text_data else f"{video_id}.npy"
            text_feat = text_data[text_key].astype(np.float32)
            
        video_steps = self.annotations[video_id]['steps']
        
        # Dynamic Video-Level Edge Index Remapping
        step_id_to_local_idx = {int(s['step_id']): i for i, s in enumerate(video_steps)}
        
        raw_edges = self.recipe_edges.get(recipe_prefix, [])
        remapped_edges = []
        
        for src, dst in raw_edges:
            if src in step_id_to_local_idx and dst in step_id_to_local_idx:
                remapped_edges.append([
                    step_id_to_local_idx[src], 
                    step_id_to_local_idx[dst]
                ])
                
        if remapped_edges:
            edge_index = torch.tensor(remapped_edges, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        has_error = any(step.get('has_errors', False) for step in video_steps)
        label = 1.0 if has_error else 0.0
        
        return {
            "video_id": video_id,
            "visual_features": torch.tensor(vis_feat),
            "text_features": torch.tensor(text_feat),
            "edge_index": edge_index,
            "label": torch.tensor(label, dtype=torch.float32)
        }

def graph_collate_fn(batch):
    video_ids = [item["video_id"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    edge_indices = [item["edge_index"] for item in batch]
    
    #pad visual representations -> [B, Max_N_Steps, 768]
    visual_tensors = [item["visual_features"] for item in batch]
    max_vis_len = max(v.size(0) for v in visual_tensors)
    visual_dim = visual_tensors[0].size(1)
    
    batched_visual = torch.zeros(len(batch), max_vis_len, visual_dim)
    visual_mask = torch.zeros(len(batch), max_vis_len, dtype=torch.float32)
    for i, v in enumerate(visual_tensors):
        batched_visual[i, :v.size(0), :] = v
        visual_mask[i, :v.size(0)] = 1.0
        
    #pad textual node representations -> [B, Max_M_Nodes, 256]
    text_tensors = [item["text_features"] for item in batch]
    max_text_len = max(t.size(0) for t in text_tensors)
    text_dim = text_tensors[0].size(1)
    
    batched_text = torch.zeros(len(batch), max_text_len, text_dim)
    text_mask = torch.zeros(len(batch), max_text_len, dtype=torch.float32)
    for i, t in enumerate(text_tensors):
        batched_text[i, :t.size(0), :] = t
        text_mask[i, :t.size(0)] = 1.0
        
    return {
        "video_ids": video_ids,
        "visual_features": batched_visual,
        "text_features": batched_text,
        "visual_mask": visual_mask,
        "text_mask": text_mask,
        "edge_indices": edge_indices,
        "labels": labels
    }
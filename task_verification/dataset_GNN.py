import os

import numpy as np
import json
import torch
import zipfile
from torch.utils.data import Dataset

#official mapping between video prefix and CaptainCook4D task graph JSON files (annotations/task_graphs/)
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
        Caches DAG topologies and coordinates visual/fine-grained text representations.
        """
        self.visual_features = np.load(visual_npz_path)
        self.text_features = np.load(text_npz_path)
        self.split = split
        
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
            
        #filter valid video entries present in both visual and pre-extracted textual archives
        self.video_list = []
        for vid in video_ids:
            if vid in self.visual_features and vid in self.text_features:
                self.video_list.append(vid)
                
        #cache topologies to avoid runtime file system overhead
        self.graphs_topology = {}
        if os.path.isdir(graph_zip_path):
            for prefix, filename in RECIPE_MAPPING.items():
                file_path = os.path.join(graph_zip_path, filename)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        edges = data.get('edges', [])
                        if edges:
                            # Parse into standard directed PyG edge indexes [2, Num_Edges]
                            self.graphs_topology[prefix] = torch.tensor(edges, dtype=torch.long).t()
                        else:
                            self.graphs_topology[prefix] = torch.empty((2, 0), dtype=torch.long)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_id = self.video_list[idx]
        recipe_prefix = video_id.split('_')[0]
        
        #load raw visual frame-level step embeddings [N_steps, 768]
        vis_feat = self.visual_features[video_id].astype(np.float32)
        
        #load aligned text features [M_nodes, 256]
        text_feat = self.text_features[video_id].astype(np.float32)
        
        #retrieve graph topology
        edge_index = self.graphs_topology.get(recipe_prefix, torch.empty((2, 0), dtype=torch.long))
        
        #anomaly label extraction
        has_error = any(step.get('has_errors', False) for step in self.annotations[video_id]['steps'])
        label = 1.0 if has_error else 0.0
        
        return {
            "video_id": video_id,
            "visual_features": torch.tensor(vis_feat),
            "text_features": torch.tensor(text_feat),
            "edge_index": edge_index,
            "label": torch.tensor(label, dtype=torch.float32)
        }

def graph_collate_fn(batch):
    """
    Handles collating ragged temporal dimensions and variable node counts into structured batches.
    """
    video_ids = [item["video_id"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    edge_indices = [item["edge_index"] for item in batch]
    
    #dynamic padding for visual sequences -> [B, Max_N_Steps, 768]
    visual_tensors = [item["visual_features"] for item in batch]
    max_vis_len = max(v.size(0) for v in visual_tensors)
    visual_dim = visual_tensors[0].size(1)
    
    batched_visual = torch.zeros(len(batch), max_vis_len, visual_dim)
    visual_mask = torch.zeros(len(batch), max_vis_len, dtype=torch.float32)
    for i, v in enumerate(visual_tensors):
        batched_visual[i, :v.size(0), :] = v
        visual_mask[i, :v.size(0)] = 1.0
        
    # dynamic padding for textual node matrices -> [B, Max_M_Nodes, 256]
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
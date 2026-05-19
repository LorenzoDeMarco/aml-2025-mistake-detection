import numpy as np
import json
from scipy.optimize import linear_sum_assignment

if __name__ == "__main__":
    visual_npz = 'step_embeddings_dataset.npz'
    text_npz = 'text_task_graphs.npz'
    annotations_json = 'annotations/annotation_json/complete_step_annotations.json'
    
    print("Computing Hungarian matching offline...")
    vis_data = np.load(visual_npz)
    text_data = np.load(text_npz)
    
    with open(annotations_json, 'r') as f:
        annotations = json.load(f)
        
    precomputed_matches = {}
    
    for k in vis_data.files:
        clean_k = k.replace('.npy', '')
        text_key = clean_k if clean_k in text_data else f"{clean_k}.npy"
        
        if clean_k in annotations and text_key in text_data:
            v_feat = vis_data[k].astype(np.float32)
            t_feat = text_data[text_key].astype(np.float32)
            
            num_vis = max(1, v_feat.shape[0] // 4)
            num_text = t_feat.shape[0]
            
            v_pooled = np.array([v_feat[i*4:(i+1)*4].mean(axis=0) for i in range(num_vis)])
            
            #normalize for cosine cost initialization
            v_norm = v_pooled / np.linalg.norm(v_pooled, axis=-1, keepdims=True)
            t_norm = t_feat / np.linalg.norm(t_feat, axis=-1, keepdims=True)
            
            similarity = np.dot(v_norm, t_norm.T)
            cost = 1.0 - similarity
            
            v_idx, t_idx = linear_sum_assignment(cost)
            
            #save raw assignments pairs as string representation matrix
            match_matrix = np.stack([v_idx, t_idx], axis=0)
            precomputed_matches[clean_k] = match_matrix
            
    np.savez('hungarian_matches.npz', **precomputed_matches)
    print("Hungarian matching dictionary pre-computed and stored in hungarian_matches.npz safely!")
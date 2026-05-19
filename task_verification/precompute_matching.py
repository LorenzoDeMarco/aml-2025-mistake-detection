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
        
    #set seed to generate a stable, deterministic common metric space at initialization
    rng = np.random.default_rng(42)
    joint_dim = 256
    visual_dim = 768
    text_dim = 256
    
    #create deterministic random projection matrices to align dimensions offline
    W_vis = rng.normal(0.0, 1.0 / np.sqrt(visual_dim), (visual_dim, joint_dim)).astype(np.float32)
    W_text = rng.normal(0.0, 1.0 / np.sqrt(text_dim), (text_dim, joint_dim)).astype(np.float32)
    
    precomputed_matches = {}
    
    for k in vis_data.files:
        clean_k = k.replace('.npy', '')
        text_key = clean_k if clean_k in text_data else f"{clean_k}.npy"
        
        if clean_k in annotations and text_key in text_data:
            v_feat = vis_data[k].astype(np.float32) # Shape: [N_frames, 768]
            t_feat = text_data[text_key].astype(np.float32) # Shape: [256, M_nodes] or [M_nodes, 256]
            
            if t_feat.shape[0] == text_dim and t_feat.shape[1] != text_dim:
                t_feat = t_feat.T # transpose to get [M_nodes, 256]
                
            # temporal pooling stride-4 emulation
            num_vis = max(1, v_feat.shape[0] // 4)
            num_text = t_feat.shape[0]
            
            #reconstruct average pooling execution over windows of size 4
            v_pooled = []
            for i in range(num_vis):
                window = v_feat[i*4 : (i+1)*4]
                v_pooled.append(window.mean(axis=0))
            v_pooled = np.array(v_pooled) # Shape: [Num_vis, 768]
            
            # project both modalities into the common 256-dimensional space
            proj_v = np.dot(v_pooled, W_vis) # [Num_vis, 256]
            proj_t = np.dot(t_feat, W_text)   # [M_nodes, 256]
            
            #normalize representations to compute pure cosine similarity
            v_norm = proj_v / np.linalg.norm(proj_v, axis=-1, keepdims=True)
            t_norm = proj_t / np.linalg.norm(proj_t, axis=-1, keepdims=True)
            
            # structural multiplication ->cost matrix calculation
            similarity = np.dot(v_norm, t_norm.T) # Shape: [Num_vis, M_nodes]
            cost = 1.0 - similarity
            
            v_idx, t_idx = linear_sum_assignment(cost)
            
            match_matrix = np.stack([v_idx, t_idx], axis=0)
            precomputed_matches[clean_k] = match_matrix
            
    np.savez('hungarian_matches.npz', **precomputed_matches)
    print("Hungarian matching dictionary pre-computed and stored in hungarian_matches.npz ")
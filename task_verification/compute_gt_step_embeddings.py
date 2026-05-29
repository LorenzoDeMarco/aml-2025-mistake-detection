import json
import numpy as np
import os
from tqdm import tqdm

def create_gt_step_embeddings(json_path, feat_dir, output_file, fps=1.876):
    """
    Uses Ground Truth annotations to create averaged step-level embeddings.
    
    Args:
        json_path (str): Path to the ground truth JSON file.
        feat_dir (str): Directory containing the original .npz EgoVLP features.
        output_file (str): Path where the final .npz dataset will be saved.
        fps (float): Frames per second used for feature extraction.
    """
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    # load ground truth annotations
    with open(json_path, 'r') as f:
        gt_data = json.load(f)
        
    print(f"Loaded ground truth annotations for {len(gt_data)} videos.")

    video_step_embeddings = {}
    missing_videos = []

    print(f"Processing videos...")
    for video_id, video_info in tqdm(gt_data.items()):
        feat_path = os.path.join(feat_dir, f"{video_id}.npz")
        
        if not os.path.exists(feat_path):
            missing_videos.append(video_id)
            continue
            
        # load EgoVLP features 
        try:
            npz_data = np.load(feat_path)
            video_features = npz_data['features'] if 'features' in npz_data else npz_data['arr_0']
        except Exception:
            missing_videos.append(video_id)
            continue
            
        total_feats = video_features.shape[0]
        step_embs = []
        
        #extract steps and sort them chronologically 
        steps = video_info.get('steps', [])
        steps_sorted = sorted(steps, key=lambda x: x['start_time'])
        
        for step in steps_sorted:
            #convert timestamps to feature indices
            start_idx = int(step['start_time'] * fps)
            end_idx = int(step['end_time'] * fps)
            
            #boundary clipping
            start_idx = max(0, min(start_idx, total_feats - 1))
            end_idx = max(start_idx + 1, min(end_idx, total_feats))
                
            #compute temporal average pooling for the segment
            feat_slice = video_features[start_idx:end_idx]
            
            if len(feat_slice) > 0:
                step_emb = np.mean(feat_slice, axis=0)
                step_embs.append(step_emb)
            else:
                #fallback: if the step is extremely short (< 1 frame), take the closest single frame
                safe_idx = min(start_idx, total_feats - 1)
                step_embs.append(video_features[safe_idx])
            
        if step_embs:
            #store as [Num_Steps, 768]
            video_step_embeddings[video_id] = np.array(step_embs, dtype=np.float32)

    #save the structured dataset
    np.savez(output_file, **video_step_embeddings)
    
    print(f"\nSuccessfully created {output_file}")
    print(f"Total videos processed and saved: {len(video_step_embeddings)}")
    
    if missing_videos:
        print(f"WARNING: Missing feature files for {len(missing_videos)} videos.")

if __name__ == "__main__":
    create_gt_step_embeddings(
        json_path="annotations/annotation_json/complete_step_annotations.json",
        feat_dir="./data/egovlp_features",
        output_file="gt_step_embeddings.npz"
    )
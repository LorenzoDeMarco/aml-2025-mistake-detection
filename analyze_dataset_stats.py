import json
import numpy as np
import argparse

def analyze_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    database = data.get('database', {})
    
    durations = []
    video_lengths = []
    
    for video_id, video_info in database.items():
        # Get video length 
        vid_duration = video_info.get('duration', 0)
        video_lengths.append(vid_duration)
        
        # Get duration of each action step
        annotations = video_info.get('annotations', [])
        for ann in annotations:
            segment = ann.get('segment', [0, 0])
            duration = segment[1] - segment[0]
            durations.append(duration)
            
    durations = np.array(durations)
    video_lengths = np.array(video_lengths)
    
    print("=== CaptainCook4D Dataset Statistics ===")
    print(f"Total Videos: {len(video_lengths)}")
    print(f"Total Action Steps: {len(durations)}")
    print("-" * 40)
    print("--- Video Lengths (For max_seq_len) ---")
    print(f"Max Video Length: {np.max(video_lengths):.2f}s")
    print(f"Avg Video Length: {np.mean(video_lengths):.2f}s")
    print("-" * 40)
    print("--- Action Durations (For regression_range) ---")
    print(f"Min Action Length: {np.min(durations):.2f}s")
    print(f"Max Action Length: {np.max(durations):.2f}s")
    print(f"Avg Action Length: {np.mean(durations):.2f}s")
    print(f"Percentiles (20%, 40%, 60%, 80%):")
    print(np.percentile(durations, [20, 40, 60, 80]))
    print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True, 
                        help="Path to actionformer_format.json")
    args = parser.parse_args()
    
    analyze_dataset(args.json_path)
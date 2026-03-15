import os
import sys
import json
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from tqdm import tqdm 
import pathlib

# Paths Configuration
BACKBONE_NAME = "egovlp"
VIDEO_DIR = "data/raw_videos" 
OUTPUT_DIR = f"data/video/{BACKBONE_NAME}"
WEIGHTS_PATH = "feature_extraction/egovlp.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_backbone():
    sys.path.append("EgoVLP") 
    from model.model import FrozenInTime
    
    config_path = os.path.join("EgoVLP", "configs", "pt", "egoclip.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    model = FrozenInTime(**config['arch']['args'])
    
    print("Loading EgoVLP weights...")
    
    temp_posix = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    
    pathlib.PosixPath = temp_posix
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model.eval()
    model.to(device)
    
    return model
def preprocess_frames(frames):
    # Frames from decord are in (Time, Height, Width, Channels) uint8 format
    frames = frames.float() / 255.0
    
    # Reshape to (Time, Channels, Height, Width) to apply torchvision transformations
    frames = frames.permute(0, 3, 1, 2)
    
    # Official EgoVLP transformation pipeline (TimeSformer)
    transform = T.Compose([
        T.Resize(256, antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations -> Shape: (T, C, 224, 224)
    frames = transform(frames)
    
    # The model expects standard video dimensionality (Batch, Channels, Time, Height, Width)
    frames = frames.unsqueeze(0)        # Add batch dimension: (1, C, T, 224, 224)
    
    return frames.to(device)

def process_video(video_path, model):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    duration_sec = int(total_frames / fps)
    
    # ACTIONFORMER POLICY (dense extraction)
    WINDOW_SIZE = 16 
    STRIDE = 16       
    
    features_per_sec = {sec: [] for sec in range(duration_sec + 1)}
    
    with torch.no_grad():
        for start_frame in range(0, total_frames - WINDOW_SIZE + 1, STRIDE):
            end_frame = start_frame + WINDOW_SIZE
            center_frame = start_frame + (WINDOW_SIZE // 2)
            sec_idx = min(int(center_frame / fps), duration_sec)
            
            frame_indices = list(range(start_frame, end_frame))
            batch = vr.get_batch(frame_indices)
            
            frames = batch if isinstance(batch, torch.Tensor) else torch.from_numpy(batch.asnumpy())
            
            input_tensor = preprocess_frames(frames)
            
            # In Dual-Encoder models like FrozenInTime, we only need to compute video features
            # The "compute_video" method performs the forward pass only on the visual encoder
            if hasattr(model, 'compute_video'):
                features = model.compute_video(input_tensor)
            else:
                # Fallback in case the implementation uses a conditional forward pass
                output = model(video=input_tensor)
                features = output['video_embeds'] if isinstance(output, dict) else output
                
            features = features.squeeze().cpu().numpy()
            features_per_sec[sec_idx].append(features)
            
    aggregated_features = []
    feature_dim = None

    for sec in range(duration_sec):
        if len(features_per_sec[sec]) > 0:
            sec_feature = np.mean(features_per_sec[sec], axis=0)
            if feature_dim is None:
                feature_dim = sec_feature.shape
        else:
            if feature_dim is not None:
                sec_feature = np.zeros(feature_dim)
            else:
                continue
        aggregated_features.append(sec_feature)
        
    return np.vstack(aggregated_features) if aggregated_features else np.empty((0, 0))

def main():
    model = load_backbone()
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    
    for video_filename in tqdm(video_files, desc="EgoVLP Extraction"):
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        recording_id = video_filename.replace('.mp4', '').replace('_360p', '') 
        output_filename = f"{recording_id}_360p.mp4_1s_1s.npz"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        if os.path.exists(output_path):
            continue
            
        features_matrix = process_video(video_path, model)
        if features_matrix.size > 0:
            np.savez(output_path, arr_0=features_matrix)

if __name__ == "__main__":
    main()  
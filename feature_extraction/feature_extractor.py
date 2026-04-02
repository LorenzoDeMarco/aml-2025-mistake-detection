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
    
    if os.name == 'nt':
        temp_posix = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    
    if os.name == 'nt':
        pathlib.PosixPath = temp_posix
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    if hasattr(model, 'video_projection'):
        model.video_projection = torch.nn.Identity()
    
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

def process_video(video_path, model, target_hz=1.875, window_size=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    
    # Dynamic stride calculation based on target frequency and native FPS
    stride = int(round(fps / target_hz))
    
    # Safety check to prevent infinite loops if the target frequency is exceptionally high
    if stride < 1:
        stride = 1
        
    aggregated_features = []
    
    with torch.no_grad():
        for start_frame in range(0, total_frames - window_size + 1, stride):
            end_frame = start_frame + window_size
            frame_indices = list(range(start_frame, end_frame))
            
            batch = vr.get_batch(frame_indices)
            frames = batch if isinstance(batch, torch.Tensor) else torch.from_numpy(batch.asnumpy())
            
            input_tensor = preprocess_frames(frames)
            
            if hasattr(model, 'compute_video'):
                features = model.compute_video(input_tensor)
            else:
                output = model(video=input_tensor)
                features = output['video_embeds'] if isinstance(output, dict) else output
                
            features = features.squeeze().cpu().numpy()
            aggregated_features.append(features)
            
    return np.vstack(aggregated_features) if aggregated_features else np.empty((0, 0))

def main():
    model = load_backbone()
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    
    # --- EXTRACTION CONFIGURATION ---
    TARGET_HZ = 1.875
    WINDOW_SIZE = 16
    # --------------------------------
    
    for video_filename in tqdm(video_files, desc="EgoVLP Feature Extraction"):
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        recording_id = video_filename.replace('.mp4', '').replace('_360p', '') 
        
        # Generate the output filename reflecting the chosen frequency
        output_filename = f"{recording_id}_360p.mp4_{TARGET_HZ}hz.npz"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        if os.path.exists(output_path):
            continue
            
        features_matrix = process_video(video_path, model, target_hz=TARGET_HZ, window_size=WINDOW_SIZE)
        
        if features_matrix.size > 0:
            np.savez(output_path, arr_0=features_matrix)

if __name__ == "__main__":
    main()
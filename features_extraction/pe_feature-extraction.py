import os
import numpy as np
import torch
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

# Perception Encoder imports (from facebookresearch/perception_models)
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# ---------------------------------------------------------------------------
# Paths Configuration
# ---------------------------------------------------------------------------
BACKBONE_NAME = "perception_encoder"
VIDEO_DIR     = "data/raw_videos"
OUTPUT_DIR    = f"data/video/{BACKBONE_NAME}"

# PE model config — scegli tra:
#   "PE-Core-G14-448", "PE-Core-L14-336", "PE-Core-B16-224",
#   "PE-Core-S16-384", "PE-Core-T16-384"
PE_CONFIG = "PE-Core-L14-336"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_backbone():
    print(f"Loading Perception Encoder ({PE_CONFIG})...")
    model = pe.CLIP.from_config(PE_CONFIG, pretrained=True)  # scarica da HuggingFace
    model.eval()
    model.to(device)

    # transform ufficiale per immagini (applicheremo frame per frame)
    preprocess = transforms.get_image_transform(model.image_size)
    return model, preprocess


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------
def process_video(
    video_path: str,
    model,
    preprocess,
    window_sec: float = 8.0,
    stride_sec: float = 8.0,
    num_frames: int   = 8,
):
    """
    Estrae un embedding per ogni finestra temporale del video.

    Per ogni finestra di `window_sec` secondi (con passo `stride_sec`),
    campiona `num_frames` frame uniformemente distribuiti all'interno della
    finestra, li processa con PE e ne fa average pooling — esattamente come
    descritto nel paper PE (§2.2, §2.4).

    Args:
        window_sec:  durata di ogni finestra in secondi (es. 8.0)
        stride_sec:  passo tra una finestra e la successiva in secondi.
                     Se uguale a window_sec le finestre non si sovrappongono.
        num_frames:  frame uniformi da campionare per finestra (default 8,
                     come nel paper PE)

    Returns:
        np.ndarray di shape (num_windows, embedding_dim)
    """
    decoder      = VideoDecoder(video_path)
    metadata     = decoder.metadata
    fps          = float(metadata.average_fps)
    total_frames = metadata.num_frames

    window_frames = max(num_frames, int(round(window_sec * fps)))
    stride_frames = max(1,          int(round(stride_sec * fps)))

    aggregated_features = []

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for start_frame in range(0, total_frames - window_frames + 1, stride_frames):
            end_frame = start_frame + window_frames

            # Campiona `num_frames` indici uniformemente nella finestra
            indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int).tolist()

            # torchcodec → (T, C, H, W) float in [0, 1]
            frames = decoder.get_frames_at(indices=indices).data

            # Preprocess frame per frame e stack → (T, C, H, W)
            processed = torch.stack([
                preprocess(frame)
                for frame in frames
            ]).to(device)

            # PE image encoder: ogni frame → (D,), poi average pooling → (D,)
            frame_features = []
            for frame_tensor in processed:
                feat, _, _ = model(frame_tensor.unsqueeze(0), None)  # image branch
                frame_features.append(feat.squeeze(0))

            window_feat = torch.stack(frame_features).mean(dim=0)
            aggregated_features.append(window_feat.float().cpu().numpy())

    return np.vstack(aggregated_features) if aggregated_features else np.empty((0, 0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    model, preprocess = load_backbone()

    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

    # --- EXTRACTION CONFIGURATION ---
    WINDOW_SEC = 8.0   # durata finestra in secondi
    STRIDE_SEC = 8.0   # passo tra finestre (= WINDOW_SEC → no overlap)
    NUM_FRAMES = 8     # frame uniformi per finestra (come nel paper PE)
    # --------------------------------

    for video_filename in tqdm(video_files, desc="PE Feature Extraction"):
        video_path = os.path.join(VIDEO_DIR, video_filename)

        recording_id    = video_filename.replace(".mp4", "").replace("_360p", "")
        output_filename = f"{recording_id}_360p.mp4_{WINDOW_SEC}s.npz"
        output_path     = os.path.join(OUTPUT_DIR, output_filename)

        if os.path.exists(output_path):
            continue

        features_matrix = process_video(
            video_path, model, preprocess,
            window_sec=WINDOW_SEC,
            stride_sec=STRIDE_SEC,
            num_frames=NUM_FRAMES,
        )

        if features_matrix.size > 0:
            np.savez(output_path, arr_0=features_matrix)


if __name__ == "__main__":
    main()
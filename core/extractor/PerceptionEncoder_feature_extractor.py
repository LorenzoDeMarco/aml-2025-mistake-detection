import argparse
import os
import numpy as np
import torch
import concurrent.futures
import logging
from contextlib import nullcontext
from tqdm import tqdm
from PIL import Image

# pytorchvideo backbones need these transforms
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
import torchvision.transforms as T
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo

# your local omnivore transforms
from omnivore_transforms import SpatialCrop, TemporalCrop

# optional: fallback decode (slow)
from pytorchvideo.data.encoded_video import EncodedVideo

# optional: fast decode (open once)
try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader = None
    cpu = None


# -------------------------
# Args
# -------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Segment feature extractor (fast decode, multiple backbones).")

    parser.add_argument("--backbone", type=str, default="omnivore",
                        help="Options: omnivore, slowfast, x3d, 3dresnet, peav, pe_core")

    parser.add_argument("--video_dir", type=str,
                        default=r"./data/video",
                        help="Directory containing .mp4 videos")
    parser.add_argument("--output_dir", type=str,
                        default=r"./data/features",
                        help="Base output directory (a subfolder per backbone will be created)")

    parser.add_argument("--segment_seconds", type=float, default=5.0,
                        help="Segment length in seconds (bigger = fewer segments, faster).")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel videos to process (keep 1 for small GPUs).")

    parser.add_argument("--use_decord", action="store_true", default=True,
                        help="Use decord open-once decoding if available (recommended).")
    parser.add_argument("--no_decord", action="store_true", default=False,
                        help="Force disable decord even if installed (debug).")

    # --- PE-AV (heavy, kept for compatibility) ---
    parser.add_argument("--peav_model", type=str, default="facebook/pe-av-base-16-frame",
                        help="HF model id for PE-AV (heavy).")
    parser.add_argument("--peav_num_frames", type=int, default=16,
                        help="Uniform frames sampled per segment for PE-AV.")
    parser.add_argument("--peav_dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"],
                        help="Autocast dtype for PE-AV on CUDA (GTX1050: fp16 recommended).")

    # --- PE-Core (light, recommended) ---
    parser.add_argument("--pe_core_model_id", type=str, default="hf-hub:timm/PE-Core-B-16",
                        help="OpenCLIP model id for PE-Core, e.g. hf-hub:timm/PE-Core-B-16")
    parser.add_argument("--pe_core_num_frames", type=int, default=8,
                        help="Uniform frames sampled per segment for PE-Core.")
    parser.add_argument("--pe_core_dtype", type=str, default="fp16", choices=["fp16", "fp32"],
                        help="Autocast dtype for PE-Core on CUDA.")
    parser.add_argument("--pe_core_batch_size", type=int, default=32,
                        help="Batch size when encoding frames for PE-Core.")

    return parser.parse_args()


# -------------------------
# Utilities
# -------------------------
def _ensure_uint8_video(x: torch.Tensor) -> torch.Tensor:
    # x: (C,T,H,W)
    if x.dtype != torch.uint8:
        x = x.clamp(0, 255).to(torch.uint8)
    return x


def _tensor_video_to_pil_list(video: torch.Tensor):
    """
    video: (C, T, H, W) torch tensor
    return: List[PIL.Image] length T, RGB
    """
    video = _ensure_uint8_video(video)
    v = video.permute(1, 2, 3, 0)  # (T,H,W,C)
    v = v.cpu().numpy()

    frames = []
    for frame in v:
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        elif frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        img = Image.fromarray(frame).convert("RGB")
        frames.append(img)
    return frames


def _frames_needed_for_backbone(method: str, args) -> int:
    m = method.lower()
    if m in ["omnivore", "slowfast"]:
        return 32
    if m in ["x3d"]:
        return 16
    if m in ["3dresnet"]:
        return 8
    if m in ["peav", "pe-av", "pe_av"]:
        return int(args.peav_num_frames)
    if m in ["pe_core", "pe-core", "pe_core_clip","perception_encoder"]:
        return int(args.pe_core_num_frames)
    return 16


def _linspace_indices(start: int, end: int, n: int) -> np.ndarray:
    if end <= start:
        return np.array([start], dtype=np.int64)
    if n <= 1:
        return np.array([start], dtype=np.int64)
    idx = np.linspace(start, end - 1, n)
    idx = np.clip(np.round(idx), start, end - 1).astype(np.int64)
    return idx


# -------------------------
# PE-AV Extractor (kept, heavy)
# -------------------------
class PEAVExtractor(torch.nn.Module):
    """
    Video-only features via PE-AV.
    Note: PE-AV is heavy; PE-Core is recommended for GTX1050.
    """
    def __init__(self, model_name: str, device: torch.device, autocast_dtype: str = "fp16"):
        super().__init__()
        try:
            from transformers import PeAudioVideoModel, PeAudioVideoProcessor
        except Exception as e:
            raise ImportError(
                "Cannot import PeAudioVideoModel/Processor. Install a newer transformers:\n"
                "  pip install -U git+https://github.com/huggingface/transformers\n"
            ) from e

        self.device = device
        self.autocast_dtype = autocast_dtype
        self.model = PeAudioVideoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = PeAudioVideoProcessor.from_pretrained(model_name)

    def _autocast_ctx(self):
        if self.device.type != "cuda":
            return nullcontext()
        if self.autocast_dtype == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if self.autocast_dtype == "fp16":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    @torch.inference_mode()
    def forward(self, frames_pil_list):
        frames_pil_list = [im.convert("RGB") for im in frames_pil_list]
        inputs = self.processor(videos=[frames_pil_list], return_tensors="pt", padding=True)
        pixel_values_videos = inputs["pixel_values_videos"].to(self.device)
        padding_mask_videos = inputs.get("padding_mask_videos", None)
        if padding_mask_videos is not None:
            padding_mask_videos = padding_mask_videos.to(self.device)

        with self._autocast_ctx():
            # Most stable across transformers versions: call the internal video model features
            if hasattr(self.model, "video_model") and hasattr(self.model.video_model, "get_video_features"):
                feats = self.model.video_model.get_video_features(
                    pixel_values_videos=pixel_values_videos,
                    padding_mask_videos=padding_mask_videos,
                )
            else:
                raise RuntimeError("PeVideoModel.get_video_features not found; upgrade transformers.")

            if feats.ndim == 3:
                feats = feats.mean(dim=1)  # (B,D)

        return feats.float()  # (B,D), float32 for safe numpy save


# -------------------------
# PE-Core Extractor (light, recommended)
# -------------------------
class PECoreExtractor(torch.nn.Module):
    """
    Light Perception Encoder Core (OpenCLIP style).
    Input: frames_pil_list: List[PIL.Image]
    Output: (1, D) tensor
    """
    def __init__(self, model_id: str, device: torch.device, autocast_dtype: str = "fp16", batch_size: int = 32):
        super().__init__()
        try:
            import open_clip
        except Exception as e:
            raise ImportError("Please install open_clip_torch and timm:\n  pip install -U open_clip_torch timm") from e

        self.device = device
        self.autocast_dtype = autocast_dtype
        self.batch_size = batch_size

        model, _, preprocess = open_clip.create_model_and_transforms(model_id)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess

    def _autocast_ctx(self):
        if self.device.type != "cuda":
            return nullcontext()
        if self.autocast_dtype == "fp16":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    @torch.inference_mode()
    def forward(self, frames_pil_list):
        frames_pil_list = [im.convert("RGB") for im in frames_pil_list]
        imgs = [self.preprocess(im) for im in frames_pil_list]  # each: (C,H,W)
        x = torch.stack(imgs, dim=0).to(self.device)            # (T,C,H,W)

        feats = []
        with self._autocast_ctx():
            for i in range(0, x.shape[0], self.batch_size):
                xb = x[i:i + self.batch_size]
                fb = self.model.encode_image(xb, normalize=True)  # (b,D)
                feats.append(fb)

        feats = torch.cat(feats, dim=0)      # (T,D)
        seg_feat = feats.mean(dim=0)         # (D,)
        return seg_feat.unsqueeze(0).float() # (1,D) float32


# -------------------------
# Transforms
# -------------------------
def get_video_transformation(name, args):
    name = name.lower()

    if name == "omnivore":
        num_frames = 32
        video_transform = T.Compose(
            [
                UniformTemporalSubsample(num_frames),
                T.Lambda(lambda x: x / 255.0),
                ShortSideScale(size=224),
                NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                TemporalCrop(frames_per_clip=32, stride=40),
                SpatialCrop(crop_size=224, num_crops=3),
            ]
        )
        return ApplyTransformToKey(key="video", transform=video_transform)

    if name == "slowfast":
        slowfast_alpha = 4
        num_frames = 32
        side_size = 256
        crop_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]

        class PackPathway(torch.nn.Module):
            def forward(self, frames: torch.Tensor):
                fast_pathway = frames
                slow_pathway = torch.index_select(
                    frames,
                    1,
                    torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha).long(),
                )
                return [slow_pathway, fast_pathway]

        video_transform = T.Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size),
                PackPathway(),
            ]
        )
        return ApplyTransformToKey(key="video", transform=video_transform)

    if name == "x3d":
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        side_size = 256
        crop_size = 256
        num_frames = 16

        video_transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size=(crop_size, crop_size)),
            ]
        )
        return ApplyTransformToKey(key="video", transform=video_transform)

    if name == "3dresnet":
        side_size = 256
        crop_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        num_frames = 8

        video_transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size=(crop_size, crop_size)),
            ]
        )
        return ApplyTransformToKey(key="video", transform=video_transform)

    if name in ["peav", "pe-av", "pe_av"]:
        video_transform = Compose(
            [
                UniformTemporalSubsample(int(args.peav_num_frames)),
                Lambda(_tensor_video_to_pil_list),
            ]
        )
        return ApplyTransformToKey(key="video", transform=video_transform)

    if name in ["pe_core", "pe-core", "pe_core_clip","perception_encoder"]:
        video_transform = Compose(
            [
                UniformTemporalSubsample(int(args.pe_core_num_frames)),
                Lambda(_tensor_video_to_pil_list),
            ]
        )
        return ApplyTransformToKey(key="video", transform=video_transform)

    raise ValueError(f"Unknown backbone: {name}")


# -------------------------
# Models
# -------------------------
def get_feature_extractor(name, device, args):
    name = name.lower()

    if name == "omnivore":
        model_name = "omnivore_swinB_epic"
        model = torch.hub.load("facebookresearch/omnivore:main", model=model_name)
        model.heads = torch.nn.Identity()
        return model.to(device).eval()

    if name == "slowfast":
        model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
        model.heads = torch.nn.Identity()
        return model.to(device).eval()

    if name == "x3d":
        model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)
        model.heads = torch.nn.Identity()
        return model.to(device).eval()

    if name == "3dresnet":
        model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
        model.heads = torch.nn.Identity()
        return model.to(device).eval()

    if name in ["peav", "pe-av", "pe_av"]:
        return PEAVExtractor(model_name=args.peav_model, device=device, autocast_dtype=args.peav_dtype).to(device).eval()

    if name in ["pe_core", "pe-core", "pe_core_clip","perception_encoder"]:
        return PECoreExtractor(
            model_id=args.pe_core_model_id,
            device=device,
            autocast_dtype=args.pe_core_dtype,
            batch_size=args.pe_core_batch_size
        ).to(device).eval()

    raise ValueError(f"Unknown backbone: {name}")


# -------------------------
# Feature extraction wrapper
# -------------------------
def extract_features(video_data_raw, feature_extractor, transforms_to_apply, method: str, device: torch.device):
    # video_data_raw: torch (C,T,H,W) uint8
    video_data_for_transform = {"video": video_data_raw, "audio": None}
    video_data = transforms_to_apply(video_data_for_transform)
    video_inputs = video_data["video"]

    m = method.lower()

    if m == "omnivore":
        # omnivore transforms produce list of crops; take first crop
        video_input = video_inputs[0][None, ...].to(device)
        with torch.no_grad():
            feats = feature_extractor(video_input)
        feats = feats.float()
        return feats.cpu().numpy()

    if m == "slowfast":
        video_input = [i.to(device)[None, ...] for i in video_inputs]
        with torch.no_grad():
            feats = feature_extractor(video_input)
        feats = feats.float()
        return feats.cpu().numpy()

    if m in ["x3d", "3dresnet"]:
        video_input = video_inputs.unsqueeze(0).to(device)
        with torch.no_grad():
            feats = feature_extractor(video_input)
        feats = feats.float()
        return feats.cpu().numpy()

    if m in ["peav", "pe-av", "pe_av", "pe_core", "pe-core", "pe_core_clip","perception_encoder"]:
        # list[PIL.Image]
        video_input = video_inputs
        with torch.no_grad():
            feats = feature_extractor(video_input)  # expected (1,D)
        feats = feats.float()
        return feats.cpu().numpy()

    raise ValueError(f"Unknown method: {method}")


# -------------------------
# Video processing (fast decode open once)
# -------------------------
class VideoProcessor:
    def __init__(self, method, feature_extractor, video_transform, args, device):
        self.method = method
        self.feature_extractor = feature_extractor
        self.video_transform = video_transform
        self.args = args
        self.device = device

    def process_video(self, video_name, video_directory_path, output_features_path):
        segment_size_sec = float(self.args.segment_seconds)
        stride = 1  # keep naming compatible with older code

        video_path = os.path.join(
            video_directory_path,
            f"{video_name}.mp4" if "mp4" not in video_name.lower() else video_name
        )

        os.makedirs(output_features_path, exist_ok=True)
        output_file_path = os.path.join(output_features_path, video_name)

        out_npz = f"{output_file_path}_{int(segment_size_sec)}s_{int(stride)}s.npz"
        if os.path.exists(out_npz):
            logger.info(f"Skipping video: {video_name}")
            return

        use_decord = (self.args.use_decord and not self.args.no_decord and VideoReader is not None)

        # ---------- FAST PATH: decord open once ----------
        if use_decord:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            real_fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
            total_frames = len(vr)
            video_duration = total_frames / real_fps

            logger.info(f"[decord] video: {video_name} duration={video_duration:.2f}s fps={real_fps:.2f} frames={total_frames}")

            decode_frames = _frames_needed_for_backbone(self.method, self.args)
            seg_len_frames = max(int(round(segment_size_sec * real_fps)), 1)
            num_segments = int(np.ceil(total_frames / seg_len_frames))

            video_features = []
            for seg_idx in tqdm(range(num_segments), desc=f"Processing video segments for video {video_name}"):
                start_f = seg_idx * seg_len_frames
                end_f = min(start_f + seg_len_frames, total_frames)
                if end_f - start_f < 2:
                    continue

                idx = _linspace_indices(start_f, end_f, decode_frames)

                # decord: (T,H,W,3) uint8 RGB
                frames = vr.get_batch(idx).asnumpy()
                # to torch: (C,T,H,W)
                segment_video_inputs = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()

                seg_feat = extract_features(
                    video_data_raw=segment_video_inputs,
                    feature_extractor=self.feature_extractor,
                    transforms_to_apply=self.video_transform,
                    method=self.method,
                    device=self.device
                )
                video_features.append(seg_feat)

            if len(video_features) == 0:
                logger.warning(f"[decord] No segments extracted for video: {video_name}")
                return

            video_features = np.vstack(video_features)
            np.savez(out_npz, video_features)
            logger.info(f"[decord] Finished extraction and saving video: {video_name} video_features: {video_features.shape}")
            return

        # ---------- FALLBACK: EncodedVideo (slow, seeks a lot) ----------
        video = EncodedVideo.from_path(video_path)
        video_duration = float(video.duration)

        logger.info(f"[encodedvideo] video: {video_name} duration={video_duration:.2f}s")
        segment_end = max(video_duration - segment_size_sec + 1, 1)

        video_features = []
        for start_time in tqdm(np.arange(0, segment_end, segment_size_sec),
                               desc=f"Processing video segments for video {video_name}"):
            end_time = min(start_time + segment_size_sec, video_duration)
            if end_time - start_time < 0.04:
                continue

            video_data = video.get_clip(start_sec=float(start_time), end_sec=float(end_time))
            segment_video_inputs = video_data["video"]

            seg_feat = extract_features(
                video_data_raw=segment_video_inputs,
                feature_extractor=self.feature_extractor,
                transforms_to_apply=self.video_transform,
                method=self.method,
                device=self.device
            )
            video_features.append(seg_feat)

        if len(video_features) == 0:
            logger.warning(f"[encodedvideo] No segments extracted for video: {video_name}")
            return

        video_features = np.vstack(video_features)
        np.savez(out_npz, video_features)
        logger.info(f"[encodedvideo] Finished extraction and saving video: {video_name} video_features: {video_features.shape}")


# -------------------------
# Main
# -------------------------
def main(args):
    # reduce CPU oversubscription (often helps on Windows)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    method = args.backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_files_path = args.video_dir
    output_features_path = os.path.join(args.output_dir, method)

    video_transform = get_video_transformation(method, args)
    feature_extractor = get_feature_extractor(method, device=device, args=args)

    processor = VideoProcessor(method, feature_extractor, video_transform, args=args, device=device)

    mp4_files = [f for f in os.listdir(video_files_path) if f.lower().endswith(".mp4")]
    mp4_files.sort()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(int(args.num_workers), 1)) as executor:
        list(tqdm(
            executor.map(lambda f: processor.process_video(f, video_files_path, output_features_path), mp4_files),
            total=len(mp4_files)
        ))


if __name__ == "__main__":
    args = parse_arguments()

    log_directory = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_directory, exist_ok=True)

    log_file_path = os.path.join(log_directory, f"{args.backbone}.log")
    logging.basicConfig(
        filename=log_file_path,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(threadName)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    main(args)

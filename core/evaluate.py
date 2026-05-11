import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from base import fetch_model, save_error_type_results, save_results, test_er_model
from constants import Constants as const
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset, collate_fn


@dataclass
class EvalConfig:
    backbone: str = "perception_encoder"
    modality: List[str] = field(default_factory=lambda: [const.VIDEO])
    phase: str = const.TEST
    segment_length: int = 1
    segment_features_directory: str = "data/"
    split: str = const.RECORDINGS_SPLIT
    test_batch_size: int = 1
    ckpt_path: Optional[str] = None
    seed: int = 1000
    device: str = "cuda"
    variant: str = const.TRANSFORMER_VARIANT
    task_name: str = const.ERROR_RECOGNITION
    model_name: Optional[str] = None

def eval_er(conf: EvalConfig, threshold: float):
    device_torch = torch.device(conf.device if torch.cuda.is_available() else "cpu")
    conf.device = str(device_torch)

    model = fetch_model(conf)
    criterion = torch.nn.BCEWithLogitsLoss()
    ckpt = Path(conf.ckpt_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(conf.ckpt_path, map_location=device_torch)
    model.load_state_dict(state)
    model.to(device_torch)
    model.eval()

    conf.return_meta = True
    test_dataset = CaptainCookStepDataset(conf, const.TEST, conf.split)
    test_loader = DataLoader(
        test_dataset,
        batch_size=conf.test_batch_size,
        collate_fn=collate_fn,
    )

    test_losses, sub_step_metrics, step_metrics, error_type_metrics = test_er_model(
        model,
        test_loader,
        criterion,
        conf.device,
        phase="test",
        step_normalization=True,
        sub_step_normalization=True,
        threshold=threshold,
        return_error_type_metrics=True,
    )
    save_results(
        conf,
        sub_step_metrics,
        step_metrics,
        step_normalization=True,
        sub_step_normalization=True,
        threshold=threshold,
    )
    save_error_type_results(
        conf,
        error_type_metrics,
        step_normalization=True,
        sub_step_normalization=True,
        threshold=threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate error recognition baseline")
    parser.add_argument(
        "--split",
        type=str,
        choices=[const.STEP_SPLIT, const.RECORDINGS_SPLIT],
        required=True,
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=[const.SLOWFAST, const.OMNIVORE, const.PERCEPTION_ENCODER],
        required=True,
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=[const.MLP_VARIANT, const.TRANSFORMER_VARIANT, const.LSTM_VARIANT],
        required=True,
    )
    parser.add_argument("--phase", type=str, choices=[const.TEST], default=const.TEST)
    parser.add_argument(
        "--modality",
        type=str,
        nargs="+",
        default=[const.VIDEO],
        choices=[const.VIDEO],
        help="Must match training; default video only",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument(
        "--segment_features_directory",
        type=str,
        default="data/",
        help="Root containing features/<backbone>/ (same as training)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    conf = EvalConfig()
    conf.split = args.split
    conf.backbone = args.backbone
    conf.variant = args.variant
    conf.phase = args.phase
    conf.modality = args.modality
    conf.ckpt_path = args.ckpt
    conf.segment_features_directory = args.segment_features_directory
    conf.device = args.device
    conf.model_name = None

    eval_er(conf, args.threshold)

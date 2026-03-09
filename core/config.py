from argparse import ArgumentParser
from constants import Constants as const
import os


class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.backbone = "omnivore"
        self.modality = "video"
        self.phase = "train"
        self.segment_length = 1

        # Use this for 1 sec video features
        self.segment_features_directory = "data/"

        self.ckpt_directory = os.path.join(os.path.dirname(__file__), "../checkpoints")
        self.split = "recordings"
        self.batch_size = 1
        self.test_batch_size = 1
        self.num_epochs = 10
        self.lr = 1e-3
        self.weight_decay = 1e-3
        self.log_interval = 5
        self.dry_run = False
        self.ckpt = None
        self.seed = 1000
        self.device = "cuda"

        self.variant = const.TRANSFORMER_VARIANT
        self.model_name = None
        self.task_name = const.ERROR_RECOGNITION
        self.error_category = None

        self.enable_wandb = True

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.save_model = True
        self.__dict__.update(self.args)
        # after applying CLI args, determine device if not explicitly set
        if self.device is None:
            # choose cuda only if it's available
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description="training code")

        # ----------------------------------------------------------------------------------------------
        # CONFIGURATION PARAMETERS
        # ----------------------------------------------------------------------------------------------

        parser.add_argument("--batch_size", type=int, default=1, help="batch size")
        parser.add_argument("--test-batch-size", type=int, default=1, help="input batch size for testing (default: 1000)")
        parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
        parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path")
        parser.add_argument("--seed", type=int, default=42, help="random seed (default: 1000)")

        parser.add_argument("--backbone", type=str, default=const.OMNIVORE, help="backbone model")
        parser.add_argument("--ckpt_directory", type=str, default="/data/rohith/captain_cook/checkpoints", help="checkpoint directory")
        parser.add_argument("--split", type=str, default=const.RECORDINGS_SPLIT, help="split")
        parser.add_argument("--variant", type=str, default=const.TRANSFORMER_VARIANT, help="variant")
        parser.add_argument("--model_name", type=str, default=None, help="model name")
        parser.add_argument("--task_name", type=str, default=const.ERROR_RECOGNITION, help="task name")
        parser.add_argument("--error_category", type=str, help="error category")
        parser.add_argument("--modality", type=str, nargs="+", default=[const.VIDEO], help="audio")
        parser.add_argument("--device", type=str, default=None, help="compute device: cuda or cpu")

        return parser

    def set_model_name(self, model_name):
        self.model_name = model_name

    def print_config(self):
        """
        Prints the configuration
        :return:
        """
        print("Configuration:")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print("\n")

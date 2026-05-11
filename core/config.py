from argparse import ArgumentParser
from constants import Constants as const


class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.backbone = "perception_encoder"
        self.modality = "video"
        self.phase = "train"
        self.segment_length = 1

        # Use this for 1 sec video features
        self.segment_features_directory = "data/"

        #self.ckpt_directory = "/data/rohith/captain_cook/checkpoints/"
        self.ckpt_directory = "./checkpoints"
        self.split = const.RECORDINGS_SPLIT
        self.batch_size = 256
        self.test_batch_size = 256
        self.num_epochs = 5
        self.lr = 1e-3
        self.weight_decay = 1e-3
        self.log_interval = 5
        self.dry_run = False
        self.ckpt = None
        self.seed = 1000
        self.device = "cuda"

        #self.variant = const.TRANSFORMER_VARIANT
        self.variant = const.TRANSFORMER_VARIANT
        self.model_name = None
        self.task_name = const.ERROR_RECOGNITION
        self.error_category = None

        self.enable_wandb = True

        # Do not store ArgumentParser on self: it is not picklable and breaks
        # DataLoader(num_workers>0) on Windows when Dataset holds config.
        _parser = self.setup_parser()
        self.args = vars(_parser.parse_args())
        self.save_model = True
        self.__dict__.update(self.args)
        if getattr(self, "no_wandb", False):
            self.enable_wandb = False

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description="training code")

        # ----------------------------------------------------------------------------------------------
        # CONFIGURATION PARAMETERS
        # ----------------------------------------------------------------------------------------------

        parser.add_argument("--batch_size", type=int, default=256, help="batch size")
        parser.add_argument("--test-batch-size", type=int, default=256, help="input batch size for testing (default: 1000)")
        parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
        parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path")
        parser.add_argument("--seed", type=int, default=42, help="random seed (default: 1000)")

        parser.add_argument("--backbone", type=str, default=const.PERCEPTION_ENCODER, help="backbone model")
        parser.add_argument("--ckpt_directory", type=str, default="./checkpoints", help="checkpoint directory")
        parser.add_argument("--split", type=str, default=const.RECORDINGS_SPLIT, help="split")
        parser.add_argument("--variant", type=str, default=const.TRANSFORMER_VARIANT, help="variant")
        parser.add_argument("--model_name", type=str, default=None, help="model name")
        parser.add_argument("--task_name", type=str, default=const.ERROR_RECOGNITION, help="task name")
        parser.add_argument("--error_category", type=str, help="error category")
        parser.add_argument("--modality", type=str, nargs="+", default=[const.VIDEO], help="audio")

        parser.add_argument(
            "--train_mode",
            type=str,
            default="sub_step",
            choices=["sub_step", "step"],
            help="sub_step: 1s-segment training (default); step: step-level sequences",
        )
        parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")

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

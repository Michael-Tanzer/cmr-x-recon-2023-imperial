import argparse
import os

from datetime import datetime

import torch

this_dir = os.path.dirname(os.path.realpath(__file__))
default_logdir = os.path.join(this_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S"))


parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
parser.add_argument("--mixed_precision", type=bool, default=False, help="Use mixed precision")
parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume from")


# Data parameters
parser.add_argument(
    "--data_dir",
    type=str,
    default="/home/mt3019/biomedia/vol/biodata/data/CMRxRecon 2023",
    help="Path to the dataset",
)
parser.add_argument(
    "--val_split",
    type=float,
    default=0.1,
    help="Fraction of the data to use for validation",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="Number of subprocesses to use for data loading",
)
parser.add_argument(
    "--acc_factor",
    type=int,
    default=4,
    help="Acceleration factor for the undersampling mask",
)
parser.add_argument("--pad_crop_slices", type=int, default=7, help="Number of slices to pad/crop to")
parser.add_argument("--pad_kspace", action="store_true", help="Pad the two k-spaces to the same size")
parser.add_argument(
    "--single_slice",
    action="store_true",
    help="Use only a single slice from the volume",
)
parser.add_argument(
    "--coils",
    # can be "single" or "multi"
    type=str,
    default="single",
    choices=["single", "multi"],
    help="Use single-coil or multi-coil data",
)


# Model parameters
parser.add_argument(
    "--model",
    type=str,
    default="unet",
    help="Type of model to use",
    choices=[
        "unet",
        "unet_deep_cascade_cnn",
        "unrolled_unet",
        "latent_transformer",
        "latent_transformer_simple",
        "restormer",
        "restormer_medium",
        "restormer_simple",
        "modl",
        "modl_3d",
        "parametric-t2",
    ],
)
# number ensemble models
parser.add_argument(
    "--latent_transformer_mode",
    type=str,
    default="none",
    help="Latent transformer mode",
    choices=[
        "none",
        "encoder-decoder",
        "encoder-only",
        "lstm",
        "attention",
        "image-encoder-decoder",
        "image-encoder-only",
        "multi-scale-lt",
    ],
)
parser.add_argument("--number_ensemble_models", type=int, default=1, help="Number of ensemble models")
parser.add_argument("--transformer_conv_dims", type=int, default=64, help="Transformer conv dims")
parser.add_argument("--transformer_ff_dims", type=int, default=256, help="Transformer ff dims")
parser.add_argument("--transformer_n_heads", type=int, default=8, help="Transformer n heads")
parser.add_argument("--transformer_n_layers", type=int, default=6, help="Transformer n layers")
parser.add_argument("--recurrent_steps", type=int, default=1, help="Recurrent steps")
parser.add_argument(
    "--modl_denoising_model",
    type=str,
    default="unet",
    help="Type of model to use for MODL denoising",
    choices=[
        "simple",
        "unet",
        "unet_deep_cascade_cnn",
        "unrolled_unet",
        "restormer",
        "restormer_medium",
        "restormer_simple",
        "latent_transformer",
    ],
)
# scheduler warmup frequency
parser.add_argument(
    "--scheduler_warmup_frequency",
    type=int,
    default=1,
    help="Scheduler warmup frequency",
)


# Training parameters
parser.add_argument("--batch_size", type=int, default=16, help="Input batch size")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument(
    "--adam_betas",
    nargs=2,
    type=float,
    default=(0.9, 0.99),
    help="Betas for the Adam optimizer",
)
parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs to perform")
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use",
)
parser.add_argument("--seed", type=int, default=999, help="Random seed")
parser.add_argument(
    "--loss_function",
    type=str,
    default="none",
    help="Loss function to use",
    choices=["mse", "mae", "none"],
)
parser.add_argument(
    "--lr_step_size",
    type=int,
    default=100,
    help="Step size for the learning rate scheduler",
)
parser.add_argument("--lr_gamma", type=float, default=0.5, help="Gamma for the learning rate scheduler")

# Logging parameters
parser.add_argument("--log_freq", type=int, default=10, help="Frequency of logging - Steps")
parser.add_argument("--val_freq", type=int, default=10, help="Frequency of validation - Epochs")
parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving checkpoints - Epochs")
parser.add_argument("--log_dir", type=str, default=default_logdir, help="Logging directory")


parser.add_argument("--tasks", nargs="+", type=str, default=["t1", "t2"], help="Tasks to perform")
parser.add_argument(
    "--image_size",
    type=lambda s: [int(item) for item in s.split(",")],
    default=[256, 256],
    help="Image size",
)
parser.add_argument("--csm_update", action="store_true", help="CSM update or not")

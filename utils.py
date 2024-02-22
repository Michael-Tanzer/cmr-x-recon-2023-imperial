from argparse import Namespace
from functools import partial
import os
import random
import numpy as np
import torch
import wandb
from typing import List

from models.baselines import UNetDeepCascadeCNN, UNet, UnrolledUNet
from models.latent_transformer import LatentTransformer
from models.multi_coil import MoDL, MoDL3D
from models.predict_t2 import ParametricT2Predict
from models.restormer.model import Restormer


def set_custom_repr_tensors():
    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def kspace_to_images(images: np.ndarray) -> np.ndarray:
    """
    Revised by Fanwen. Keep the intensity of kspace and image in the same range. Make it consistent with torch version.
    """
    images = np.fft.ifftshift(images, axes=(-2, -1))
    images = np.fft.ifft2(images, axes=(-2, -1), norm="ortho")
    images = np.fft.fftshift(images, axes=(-2, -1))
    images = np.abs(images)
    return images


def compare_configs(config_old: Namespace, config_new: Namespace) -> bool:
    """
    Compares two configs and returns True if they are equal.
    """
    c_old = vars(config_old)
    c_new = vars(config_new)

    # Changed values
    for k, v in c_old.items():
        if k in c_new and c_new[k] != v:
            print(f"{k} differs - old: {v} new: {c_new[k]}")

    # New keys
    for k, v in c_new.items():
        if k not in c_old:
            print(f"{k} is new - {v}")

    # Removed keys
    for k, v in c_old.items():
        if k not in c_new:
            print(f"{k} is removed - {v}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_model_from_name(name: str, config: Namespace) -> torch.nn.Module:
    if config.single_slice:
        in_channels = 2 * 9 + 2 * 3
        out_channels = 2 * 9 + 2 * 3
    else:
        in_channels = 2 * 9 * config.pad_crop_slices + 2 * 3 * config.pad_crop_slices
        out_channels = 2 * 9 * config.pad_crop_slices + 2 * 3 * config.pad_crop_slices

    if name == "unet":
        return partial(
            UNet,
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            features=[16, 16, 32, 64, 128, 256],
        )
    elif name == "unet_deep_cascade_cnn":
        return partial(
            UNetDeepCascadeCNN,
            n_unets=5,
            in_channels=in_channels,
            layers=[16, 16, 32, 64, 128, 256],
        )
    elif name == "unrolled_unet":
        return partial(
            UnrolledUNet,
            steps=5,
            in_channels=in_channels,
            layers=[16, 16, 32, 64, 128, 256],
        )
    elif "latent_transformer" in name.lower():
        if "simple" in name.lower():
            layers = (2, 2, 2, 2, 2)
        else:
            layers = (8, 8, 16, 32, 64, 8)
        return partial(
            LatentTransformer,
            out_channels=18 if "t1" in config.tasks[0].lower() else 6,
            image_size=config.image_size,
            layers=layers,
            bilinear=True,
            dropout_rate=0.0,
            residual=True,
            data_consistency="hard",
            transformer_mode=config.latent_transformer_mode,  # "encoder-decoder" or "encoder-only" or "attention" or "none"
        )
    elif "restormer" in name.lower():
        if "simple" in name.lower():
            num_blocks = [2, 2, 2, 2]
            num_heads = [1, 2, 4, 8]
            channels = [2, 4, 8, 16]
        elif "medium" in name.lower():
            num_blocks = [4, 6, 6, 8]
            num_heads = [1, 2, 4, 8]
            channels = [32, 64, 128, 256]
        else:
            num_blocks = [4, 6, 6, 8]
            num_heads = [1, 2, 4, 8]
            channels = [48, 96, 192, 384]

        in_out_channels = 18 if "t1" in config.tasks[0].lower() else 6

        model = partial(
            Restormer,
            num_blocks=num_blocks,
            num_heads=num_heads,
            channels=channels,
            num_refinement=4,
            expansion_factor=2.66,
            in_out_channels=in_out_channels,
            data_consistency="hard",
            steps=config.recurrent_steps,
            intermediate_image_losses=True,
            latent_transformer_mode=config.latent_transformer_mode,
            transformer_conv_dims=config.transformer_conv_dims,
            transformer_ff_dims=config.transformer_ff_dims,
            transformer_n_heads=config.transformer_n_heads,
            transformer_n_layers=config.transformer_n_layers,
        )
        return model
    elif "modl" in name.lower():
        if "3d" in name.lower():
            return partial(MoDL3D)
        else:
            return partial(
                MoDL,
                denoising_model=config.modl_denoising_model,
                name=config.tasks[0].lower(),
                csm_update=config.csm_update,
            )
    elif name == "parametric-t2":
        return partial(
            ParametricT2Predict,
        )
    else:
        raise NotImplementedError(f"Model {name} not implemented.")


def get_loss_function_from_name(name: str) -> torch.nn.Module:
    if name.lower() == "mse":
        return torch.nn.MSELoss()
    elif name.lower() == "mae":
        return torch.nn.L1Loss()
    elif name.lower() == "none":
        return lambda *args, **kwargs: 0
    else:
        raise NotImplementedError(f"Loss function {name} not implemented.")


def log_image_wandb(
    images: List[torch.Tensor],
    name: str,
    step: int,
):
    slices, height, width = images[0].shape
    new_image = torch.zeros((slices * height, width * len(images)))
    for i, image in enumerate(images):
        for j in range(slices):
            new_image[j * height : (j + 1) * height, i * width : (i + 1) * width] = image[j]

    wandb.log({name: [wandb.Image(new_image, caption=name)]}, commit=True)


from pathlib import Path

# prefix components:
space = "    "
branch = "│   "
# pointers:
tee = "├── "
last = "└── "


def tree(dir_path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    """
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir():  # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            if current_depth < max_depth:
                yield from tree(
                    path,
                    prefix=prefix + extension,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                )

import torch
import torch.nn as nn
from typing import Callable


def hard_mask(kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return mask


def hard_data_consistency(x: torch.Tensor, kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.float:
        mask = mask.bool()

    new_kspace = torch.zeros_like(kspace)
    new_kspace[mask] = kspace[mask]
    new_kspace[~mask] = x[~mask]

    return new_kspace


class DataConsistency(nn.Module):
    def __init__(
        self,
        mask_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        data_consistency_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.mask_func = mask_func
        self.data_consistency_func = data_consistency_func

    def forward(self, x: torch.Tensor, kspace: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask.ndim != kspace.ndim:
            difference = kspace.ndim - mask.ndim
            for i in range(mask.ndim):
                mask = mask.unsqueeze(1)
                difference -= 1

                if difference == 0:
                    break

        if mask.shape[1] != kspace.shape[1]:
            mask = mask.repeat(1, kspace.shape[1], 1, 1)

        mask = self.mask_func(kspace, mask)
        return self.data_consistency_func(x, kspace, mask)


class HardDataConsistency(DataConsistency):
    def __init__(self):
        super().__init__(hard_mask, hard_data_consistency)


def kspace_to_image(x: torch.Tensor):
    """
    Revised by Fanwen. Keep the intensity of kspace and image in the same range.
    See fastMRI ifft2c_new: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
    """
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )


def image_to_kspace(x: torch.Tensor):
    """
    Revised by Fanwen. Keep the intensity of kspace and image in the same range.
    See fastMRI fft2c_new: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
    """
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )


def combine_coil_img(x: torch.Tensor, axis=-3):
    """
    input: image with separate coils: torch.complex64
    output: coil combined image along axis x: torch.float
    """
    return torch.sqrt(torch.sum(torch.abs(x) ** 2, axis=axis))

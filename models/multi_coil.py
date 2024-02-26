from typing import Dict, Literal, Optional, Union
import torch
import numpy as np
from os.path import expanduser
import matplotlib.pyplot as plt
import torch.nn as nn
from models.baselines import UNet, UNetDeepCascadeCNN, UnrolledUNet
from models.restormer.model import Restormer
from models.latent_transformer import LatentTransformer
import utils
import math
import os
import utils
import wandb
from models.utils import kspace_to_image, image_to_kspace, combine_coil_img


import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

home = expanduser("~")
epsilon = 1e-5
TFeps = torch.tensor(1e-5, dtype=torch.float32)

# different from tensorflow, the 2nd channel stands for the real and imaginary part.
c2r = lambda x: torch.stack([x.real, x.imag], axis=1)
# r2c takes the second dimension of real input and converts to complex
r2c = lambda x: torch.complex(x[:, 0, ...], x[:, 1, ...])


def c2mr(input):
    """
    for coil-combined image.
    input: new_nb*nt*ny*nx: torch.complex32
    output: new_nb*((+pha)*nt)*ny*nx: torch.real32
    """
    if len(input.shape) != 4:
        raise ValueError("The input should be 4D")
    else:
        [new_batchsize, timesteps, height, width] = input.shape
        # get the real and imag part
        input2 = torch.stack([torch.real(input), torch.imag(input)], axis=1)
        input2 = input2.view(new_batchsize, 2 * timesteps, height, width)
        return input2


def mr2c(input):
    """
    for coil-combined image.
    input: new_nb*((mag+pha)*nt)*ny*nx: tf.real32
    output: new_nb*nt*ny*nx: tf.complex32
    """
    if len(input.shape) != 4:
        raise ValueError("The input should be 4D")
    else:
        [new_batchsize, combined_timesteps, height, width] = input.shape
        # convert the nt back to complex
        input = input.view(new_batchsize, 2, combined_timesteps // 2, height, width)
        # combine to complex with
        input2 = input[:, 0, ...] + 1j * input[:, 1, ...]
        return input2


def combineNbNz(input):
    """
    This compact the batchsize and slices dim together for training.
    input: batchsize, timesteps, slices, coils, height, width: torch.complex32
    output: batchsize*slices, timesteps, coils, height, width: torch.complex32
    """
    if len(input.shape) != 6:
        raise ValueError("The input should be 6D")
    else:
        [batchsize, timesteps, slices, coils, height, width] = input.shape
        output = input.permute(0, 2, 1, 3, 4, 5).view(batchsize * slices, timesteps, coils, height, width)
        return output


class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, last_layer=False):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv.weight)  # Xavier initialization
        # self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.last_layer = last_layer

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        if not self.last_layer:  # apply ReLU activation, except for the last layer
            x = self.relu(x)
        return x


class dw(nn.Module):
    """
    This function create a layer of CNN consisting of convolution, batch-norm,
    and ReLU. Last layer does not have ReLU to avoid truncating the negative
    part of the learned noise and alias patterns.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(dw, self).__init__()

        self.layer1 = CNNLayer(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = CNNLayer(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer3 = CNNLayer(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer4 = CNNLayer(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer5 = CNNLayer(64, out_channels, kernel_size=3, stride=1, padding=1, last_layer=True)
        self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x + shortcut


class Aclass:
    """
    Implementation of the A operator in the MoDL paper
    Data consistency operator using CG-SENSE
    """

    def __init__(self, mask, lam):
        # get the size of mask
        s = mask.shape
        self.nrow, self.ncol = s[-2], s[-1]
        # self.csm = csm
        self.mask = mask
        self.lam = lam
        self.SF = torch.complex(
            torch.sqrt(torch.tensor(self.nrow * self.ncol, dtype=torch.float32)), torch.tensor(0.0, dtype=torch.float32)
        )

    def myAtA(self, img, csm):
        mask = torch.fft.ifftshift(self.mask, dim=(-2, -1))
        coilImage = csm * img.unsqueeze(1).repeat(1, csm.shape[1], 1, 1)
        kspace = torch.fft.fft2(coilImage) / self.SF
        temp = kspace * mask.unsqueeze(1).repeat(1, csm.shape[1], 1, 1)
        coilImgs = torch.fft.ifft2(temp) * self.SF
        coilComb = torch.sum(coilImgs * csm.conj(), axis=1) + self.lam * img
        return coilComb


def myCG(A, rhs, csm):
    """
    Complex conjugate gradient on complex data
    """

    # rhs = r2c(rhs)
    def body(i, rTr, x, r, p):
        Ap = A.myAtA(p, csm)
        alpha = rTr / (torch.sum(p.conj() * Ap)).real.to(torch.float32)
        alpha = torch.complex(alpha, torch.tensor(0.0).to(device))
        x = x + alpha * p
        r = r - alpha * Ap
        # take the real part
        rTrNew = (torch.sum(r.conj() * r)).real.to(torch.float32)
        beta = rTrNew / rTr
        beta = torch.complex(beta, torch.tensor(0.0).to(device))
        p = r + beta * p
        # print(rTrNew.item())
        return i + 1, rTrNew, x, r, p

    # the initial values of the loop variables
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    # This should yield cast the complex to real, but no worries,
    rTr = torch.sum(r.conj() * r).real.to(torch.float32)
    loopVar = i, rTr, x, r, p

    while i < 10 and rTr > 1e-10:
        i, rTr, x, r, p = body(i, rTr, x, r, p)

    out = x
    return out


class dc(nn.Module):
    """
    data consistency layer to do CG-SENSE
    """

    def __init__(self):
        super(dc, self).__init__()

    def forward(self, rhs, csm, mask, lam1):
        Aobj = Aclass(mask, lam1)
        y = myCG(Aobj, rhs, csm)
        return y


class MoDL(nn.Module):
    def __init__(
        self,
        denoising_model: Union[
            Literal["simple", "unet", "unet_deep_cascade_cnn", "unrolled_unet", "restormer"], nn.Module
        ],
        name: str = "t1",
        csm_update: bool = False,
    ):
        super(MoDL, self).__init__()
        self.csm_update = csm_update
        self.denosing_model = denoising_model

        self.lam = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.dc = dc()
        # for the csms, combined the (real+imag)*10coils together.
        self.dw_csm = dw(in_channels=20, out_channels=20)

        if name == "t1":
            channels = 18
        elif name == "t2":
            channels = 6

        if denoising_model == "simple":
            self.model = dw(in_channels=channels, out_channels=channels)
        elif denoising_model == "unet":
            self.model = UNet(
                spatial_dims=2,
                in_channels=channels,
                out_channels=channels,
                features=(32, 32, 64, 128, 256, 32),
                data_consistency="none",
            )
        elif denoising_model == "unet_deep_cascade_cnn":
            self.model = UNetDeepCascadeCNN(
                n_unets=5,
                in_channels=channels,
                layers=(32, 32, 64, 128, 256, 32),
                data_consistency="none",
            )
        elif denoising_model == "unrolled_unet":
            self.model = UnrolledUNet(
                steps=5,
                in_channels=channels,
                layers=(32, 32, 64, 128, 256, 32),
                data_consistency="none",
            )
        elif "restormer" in denoising_model:
            if "simple" in denoising_model:
                num_blocks = [2, 2, 2, 2]
                num_heads = [1, 2, 4, 8]
                number_channels = [2, 4, 8, 16]
            elif "medium" in denoising_model:
                num_blocks = [4, 6, 6, 8]
                num_heads = [1, 2, 4, 8]
                number_channels = [32, 64, 128, 256]
            else:
                num_blocks = [4, 6, 6, 8]
                num_heads = [1, 2, 4, 8]
                number_channels = [48, 96, 192, 384]

            self.model = Restormer(
                num_blocks=num_blocks,
                num_heads=num_heads,
                channels=number_channels,
                num_refinement=4,
                expansion_factor=2.66,
                in_out_channels=channels,
                data_consistency="none",
                steps=1,
                intermediate_image_losses=False,
                latent_transformer_mode="multi-scale-lt",
            )
        elif "latent_transformer" in denoising_model:
            print("Channels:", channels)
            self.model = LatentTransformer(
                out_channels=channels,
                image_size=(256, 256),
                layers=(8, 8, 16, 32, 64, 8),
                residual=True,
                transformer_mode="multi-scale-lt",
            )
        else:
            raise ValueError("Invalid denoising model")

    def forward(
        self,
        x_kspace: torch.Tensor,
        x_kspace_mask: torch.Tensor,
        y_kspace: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if "restormer" in self.denosing_model:
            norm_factor = 1e2
        else:
            norm_factor = 1e4

        x_kspace = torch.abs(x_kspace) * norm_factor * torch.exp(1j * torch.angle(x_kspace))
        if y_kspace is not None:
            y_kspace = torch.abs(y_kspace) * norm_factor * torch.exp(1j * torch.angle(y_kspace))

        assert x_kspace.shape[2] == 1, "Only single-slice data is supported"
        x_kspace = x_kspace.squeeze(2)
        # add for testing
        if y_kspace is not None:
            y_kspace = y_kspace.squeeze(2)
        x_img = kspace_to_image(x_kspace)

        batch_size, timesteps, coils, height, width = x_kspace.shape
        # change to batch_size*timesteps, coils, height, width to extract coil sensitivities (Csm)
        x_kspace_slice = x_kspace[:, 0]
        acs = x_kspace_slice[:, :, height // 2 - 12 : height // 2 + 12, :]
        fullacs = zpad_torch(acs, dim=(batch_size, coils, height, width), axis=(0, 1, 2, 3))
        acs_img = kspace_to_image(fullacs)
        csm = inati_cmap_torch(acs_img, niter=15, thresh=1e-3, verbose=False)
        # change the csm to the original shape.
        csm_original = csm.unsqueeze(1).repeat(1, timesteps, 1, 1, 1)
        # change the x_img to the coil_combined with batch_size, timesteps, height, width
        x_img_combined_complex = torch.sum(x_img * csm_original.conj(), axis=-3)
        # update the csms with the dw network
        if self.csm_update:
            csm4update = torch.view_as_real(csm).permute(0, 4, 1, 2, 3).reshape(batch_size, 2 * coils, height, width)
            csm = torch.view_as_complex(
                self.dw_csm(csm4update).reshape(batch_size, 2, coils, height, width).permute(0, 2, 3, 4, 1).contiguous()
            )
        ########################### network to be substituted #########################################
        # c2r18 takes the 2nd dimension of complex input and converts to real batch_size, timesteps*2, height, width
        x_img_combined_real = c2mr(x_img_combined_complex)
        # input the 18 channel as input and output the complex image with batch_size, timesteps, height, width
        if self.denosing_model == "simple":
            network_out_complex = mr2c(self.model(x_img_combined_real))
        elif "transformer" in self.denosing_model:
            data = x_img_combined_complex.unsqueeze(2)
            data = image_to_kspace(data)
            network_out_complex = self.model(x_kspace=data, x_kspace_mask=None, y_kspace=None, norm=False)[
                "output"
            ].squeeze(2)
            network_out_complex = kspace_to_image(network_out_complex)
        else:
            network_out_complex = self.model(
                x_kspace=x_img_combined_complex.unsqueeze(2), x_kspace_mask=None, y_kspace=None, name=name
            )["output"].squeeze(2)

        ########################### network to be substituted #########################################
        rhs = x_img_combined_complex + self.lam * network_out_complex

        # change the rhs to the shape of batch_size*timesteps, height, width
        rhs4SENSE = rhs.view(batch_size * timesteps, height, width)
        csm_9frames = csm.unsqueeze(1).repeat(1, timesteps, 1, 1, 1)
        csm4SENSE = csm_9frames.view(batch_size * timesteps, coils, height, width)
        x_kspace_mask4CG = (
            x_kspace_mask.squeeze(1).repeat(1, timesteps, 1, 1).reshape(batch_size * timesteps, height, width)
        )
        out = self.dc(rhs4SENSE, csm4SENSE, x_kspace_mask4CG, self.lam)
        out_coil_combined_image_complex = out.view(batch_size, timesteps, height, width)
        out_multi_coil_image = out_coil_combined_image_complex.unsqueeze(2).repeat(1, 1, coils, 1, 1) * csm_9frames
        out_multi_coil_kspace = image_to_kspace(out_multi_coil_image)

        # for the coil_combined output
        if y_kspace is not None:
            y_img = kspace_to_image(y_kspace)
            y_img_combined = combine_coil_img(y_img, axis=-3)
            multi_coil_loss = F.l1_loss(out_multi_coil_image.abs(), y_img.abs())
            combined_loss = F.l1_loss(out_coil_combined_image_complex.abs(), y_img_combined)
            loss = combined_loss
        else:
            loss = 0

        # keep the dims the same as input.
        out_multi_coil_kspace = out_multi_coil_kspace.unsqueeze(2)
        out_multi_coil_kspace = (
            torch.abs(out_multi_coil_kspace) / norm_factor * torch.exp(1j * torch.angle(out_multi_coil_kspace))
        )
        return {"loss": loss, "output": out_multi_coil_kspace, "lam": self.lam, "csm": csm}


def smooth_torch3(img, box=5):
    """Smooths coil images

    :param img: Input complex images, ``[y, x] or [z, y, x]``
    :param box: Smoothing block size (default ``5``)

    :returns simg: Smoothed complex image ``[y,x] or [z,y,x]``
    """

    t_real = torch.zeros_like(img)
    t_imag = torch.zeros_like(img)

    kernel_size = (box,) * len(img.shape[2:])
    padding = tuple((k - 1) // 2 for k in kernel_size)

    t_real = F.avg_pool2d(img.real, kernel_size, stride=1, padding=padding)
    t_imag = F.avg_pool2d(img.imag, kernel_size, stride=1, padding=padding)

    simg = t_real + 1j * t_imag

    return simg


def zpad_torch(x, dim=(256, 256, 16), axis=(0, 1, 2)):
    # axes = [(0, 0) for i in range(len(x.shape))]
    axes = []
    for idx, i in enumerate(axis):
        pad1 = dim[idx] // 2 - x.shape[i] // 2
        pad2 = dim[idx] - pad1 - x.shape[i]
        # different from numpy, axes should be a whole tuple
        axes.append(pad1)
        axes.append(pad2)
    # change axes tpo tuple
    axes = tuple(axes)
    axes = axes[::-1]
    return F.pad(x, axes, mode="constant")


def inati_cmap_torch(im, smoothing=5, niter=5, thresh=1e-3, verbose=False):
    # here the input is ns*nc*ny*nx
    im = im.permute(1, 0, 2, 3)
    """ Fast, iterative coil map estimation for 2D or 3D acquisitions.

    Parameters
    ----------
    im : ndarray
        Input images, [coil, y, x] or [coil, z, y, x].
    smoothing : int or ndarray-like
        Smoothing block size(s) for the spatial axes.
    niter : int
        Maximal number of iterations to run.
    thresh : float
        Threshold on the relative coil map change required for early
        termination of iterations.  If ``thresh=0``, the threshold check
        will be skipped and all ``niter`` iterations will be performed.
    verbose : bool
        If true, progress information will be printed out at each iteration.

    Returns
    -------
    coil_map : ndarray
        Relative coil sensitivity maps, [coil, y, x] or [coil, z, y, x].
    coil_combined : ndarray
        The coil combined image volume, [y, x] or [z, y, x].

    Notes
    -----
    The implementation corresponds to the algorithm described in [1]_ and is a
    port of Gadgetron's ``coil_map_3d_Inati_Iter`` routine.

    For non-isotropic voxels it may be desirable to use non-uniform smoothing
    kernel sizes, so a length 3 array of smoothings is also supported.

    References
    ----------
    .. [1] S Inati, MS Hansen, P Kellman.  A Fast Optimal Method for Coil
        Sensitivity Estimation and Adaptive Coil Combination for Complex
        Images.  In: ISMRM proceedings; Milan, Italy; 2014; p. 4407.
    """
    if im.ndim < 3 or im.ndim > 4:
        raise ValueError("Expected 3D [ncoils, ny, nx] or 4D [ncoils, nz, ny, nx] input.")

    if im.ndim == 3:
        # pad to size 1 on z for 2D + coils case
        images_are_2D = True
        im = im.unsqueeze(1)
    else:
        images_are_2D = False

    ncha = im.shape[0]  # first dim is the coil dimension

    # calculate D_sum using PyTorch operations -- D_sum is the dim of coil
    D_sum = im.sum(dim=(1, 2, 3))
    v = 1 / torch.linalg.norm(D_sum)
    D_sum *= v
    R = 0

    for cha in range(ncha):
        R += torch.conj(D_sum[cha]) * im[cha, ...]

    eps = torch.finfo(im.real.dtype).eps * torch.abs(im).mean()
    for it in range(niter):
        if verbose:
            print("Coil map estimation: iteration %d of %d" % (it + 1, niter))
        if thresh > 0:
            prevR = R.clone()
        R = torch.conj(R)
        coil_map = im * R.unsqueeze(0)
        coil_map_conv = smooth_torch3(coil_map, smoothing)

        D = coil_map_conv * torch.conj(coil_map_conv)
        R = D.sum(dim=0)
        R = torch.sqrt(R) + eps
        R = 1 / R
        coil_map = coil_map_conv * R.unsqueeze(0)
        D = im * torch.conj(coil_map)
        R = D.sum(dim=0)
        D = coil_map * R.unsqueeze(0)
        try:
            # torch >= 1.7 required for this notation
            D_sum = D.sum(dim=(1, 2, 3))
        except:
            D_sum = im.reshape(ncha, -1).sum(dim=1)
        v = 1 / torch.linalg.norm(D_sum)
        D_sum *= v

        imT = 0
        for cha in range(ncha):
            imT += torch.conj(D_sum[cha]) * coil_map[cha, ...]
        magT = torch.abs(imT) + eps
        imT /= magT
        R = R * imT
        imT = torch.conj(imT)
        coil_map = coil_map * imT.unsqueeze(0)

        if thresh > 0:
            diffR = R - prevR
            vRatio = torch.linalg.norm(diffR) / torch.linalg.norm(R)
            if verbose:
                print("vRatio = {}".format(vRatio))
            if vRatio < thresh:
                break

    coil_combined = (im * torch.conj(coil_map)).sum(0)

    if images_are_2D:
        # remove singleton z dimension that was added for the 2D case
        coil_combined = coil_combined[0, :, :]
        coil_map = coil_map[:, 0, :, :]
    # output is ns,nc,ny,nx
    coil_map = coil_map.permute(1, 0, 2, 3)
    return coil_map  # , coil_combined


#### Leave the 3D part for future #########
class CNNLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, last_layer=False):
        super(CNNLayer3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv.weight)  # Xavier initialization
        # self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.last_layer = last_layer

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        if not self.last_layer:  # apply ReLU activation, except for the last layer
            x = self.relu(x)
        return x


class dw3D(nn.Module):
    """
    This function create a layer of CNN consisting of convolution, batch-norm,
    and ReLU. Last layer does not have ReLU to avoid truncating the negative
    part of the learned noise and alias patterns.
    """

    def __init__(self):
        super(dw3D, self).__init__()

        self.layer1 = CNNLayer3D(2, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = CNNLayer3D(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer3 = CNNLayer3D(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer4 = CNNLayer3D(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer5 = CNNLayer3D(64, 2, kernel_size=3, stride=1, padding=1, last_layer=True)
        self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x + shortcut


class Aclass3D:
    """
    Implementation of the A operator in the MoDL paper
    Data consistency operator using CG-SENSE
    """

    def __init__(self, csm, mask, lam):
        # get the size of mask
        s = mask.shape
        self.nrow, self.ncol = s[-2], s[-1]
        self.csm = csm
        self.mask = mask
        self.lam = lam
        self.SF = torch.complex(
            torch.sqrt(torch.tensor(self.nrow * self.ncol, dtype=torch.float32)), torch.tensor(0.0, dtype=torch.float32)
        )

    def myAtA(self, img):
        coilImage = self.csm * img.unsqueeze(1).repeat(1, self.csm.shape[1], 1, 1, 1)
        kspace = torch.fft.fft2(coilImage) / self.SF
        temp = kspace * self.mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(
            self.csm.shape[0], self.csm.shape[1], self.csm.shape[2], 1, 1
        )
        coilImgs = torch.fft.ifft2(temp) * self.SF
        # self.lam should be multiplied with the 1st dim of img
        # self.lam*img = torch.einsum('ijklm,ln->ijknm', img, self.lam)
        coilComb = torch.sum(coilImgs * self.csm.conj(), axis=1) + self.lam * img
        return coilComb


class dc3D(nn.Module):
    def __init__(self):
        super(dc3D, self).__init__()

    def forward(self, rhs, csm, mask, lam1):
        lam2 = torch.complex(lam1, torch.tensor(0.0).to(device))
        Aobj = Aclass3D(csm, mask, lam2)
        y = myCG(Aobj, rhs)
        return y


class MoDL3D(MoDL):
    """
    This is for
    img: nb*2 * nt* nrow * ncol;
    csm: nb*ncoil * nrow * ncol;
    mask: nrow x ncol
    """

    def __init__(self):
        super(MoDL3D, self).__init__()
        self.dw = dw3D()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)
        self.dc = dc3D()


#### Leave the 3D part for future #########

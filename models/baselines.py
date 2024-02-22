# simple U-Net model with hard data consistency layers

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, List, Union, Optional

from monai.networks.nets.basic_unet import BasicUNet as _UNet
from monai.networks.nets.basic_unet import TwoConv
from monai.networks.layers.factories import Conv

from dataobject import MappingDatapoint
from models.utils import HardDataConsistency


class UNet(_UNet):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = (
            "LeakyReLU",
            {"negative_slope": 0.1, "inplace": True},
        ),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        data_consistency: str = "hard",
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            upsample=upsample,
        )

        if features is None:
            raise ValueError("features, in_channels, and out_channels must be specified")

        del self.conv_0
        del self.final_conv

        self.t1_conv_0 = TwoConv(2, 18, features[0], act, norm, bias, dropout)
        self.t2_conv_0 = TwoConv(2, 6, features[0], act, norm, bias, dropout)

        self.t1_final_conv = Conv["conv", 2](features[5], 18, kernel_size=1)
        self.t2_final_conv = Conv["conv", 2](features[5], 6, kernel_size=1)

        if data_consistency == "hard":
            self.dc = HardDataConsistency()
        elif data_consistency == "none":
            self.dc = lambda x, y, mask: x
        else:
            raise ValueError("data_consistency must be 'hard'")

    def forward_base(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down_1(x)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x)

        return u1

    def forward_t1(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.t1_conv_0(x)
        x1 = self.forward_base(x0)
        return self.t1_final_conv(x1)

    def forward_t2(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.t2_conv_0(x)
        x1 = self.forward_base(x0)
        return self.t2_final_conv(x1)

    def forward(
        self,
        x_kspace: torch.Tensor,
        x_kspace_mask: torch.Tensor,
        y_kspace: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
    ) -> torch.Tensor:
        ret = {}

        x_stacked = torch.stack([torch.real(x_kspace), torch.imag(x_kspace)], dim=1)
        (
            batch_size,
            _,
            timepoints,
            slices,
            height,
            width,
        ) = x_stacked.shape
        x_stacked = x_stacked.view(batch_size, 2 * slices * timepoints, height, width)

        if name == "t1":
            ret["output"] = self.forward_t1(x_stacked)
            ret["output"][:, :9] = self.dc(ret["output"][:, :9], x_stacked[:, :9], x_kspace_mask)
            ret["output"][:, 9:] = self.dc(ret["output"][:, 9:], x_stacked[:, 9:], x_kspace_mask)
        elif name == "t2":
            ret["output"] = self.forward_t2(x_stacked)
            ret["output"][:, :3] = self.dc(ret["output"][:, :3], x_stacked[:, :3], x_kspace_mask)
            ret["output"][:, 3:] = self.dc(ret["output"][:, 3:], x_stacked[:, 3:], x_kspace_mask)
        else:
            raise ValueError(f"Unknown name: {name}")

        if y_kspace is not None:
            y_stacked = torch.stack([torch.real(y_kspace), torch.imag(y_kspace)], dim=1)
            y_stacked = y_stacked.view(batch_size, 2 * slices * timepoints, height, width)

            if "loss" in ret:
                ret["loss"] += F.mse_loss(ret["output"], y_stacked)
            else:
                ret["loss"] = F.mse_loss(ret["output"], y_stacked)

        ret["output"] = ret["output"].view(batch_size, 2, timepoints, slices, height, width)
        ret["output"] = ret["output"][:, 0, :, :, :, :] + 1j * ret["output"][:, 1, :, :, :, :]

        return ret


class UNetDeepCascadeCNN(nn.Module):
    """
    Deep Cascade CNN model, made with the UNet class.
    """

    def __init__(
        self,
        n_unets: int,
        in_channels: int,
        layers: List[int],
        data_consistency: str = "hard",
    ):
        super().__init__()
        self.n_unets = n_unets

        self.unets = nn.ModuleList(
            [
                UNet(
                    spatial_dims=2,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    features=layers,
                    data_consistency=data_consistency,
                )
                for _ in range(n_unets)
            ]
        )

    def forward(
        self,
        x_kspace: torch.Tensor,
        x_kspace_mask: torch.Tensor,
        y_kspace: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
    ) -> torch.Tensor:
        output = {}
        for unet in self.unets:
            out = unet(
                x_kspace=x_kspace,
                x_kspace_mask=x_kspace_mask,
                y_kspace=y_kspace,
                name=name,
            )
            if "output" in out:
                x_kspace = out["output"]
            else:
                x_kspace = out

            output["output"] = x_kspace

            if "loss" in out:
                if "loss" in output:
                    output["loss"] += out["loss"]
                else:
                    output["loss"] = out["loss"]

        return output


class UnrolledUNet(nn.Module):
    """
    Unrolled UNet model, made with the UNet class.
    """

    def __init__(
        self,
        steps: int,
        in_channels: int,
        layers: List[int],
        data_consistency: str = "hard",
    ):
        super().__init__()
        self.steps = steps

        self.unet = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=in_channels,
            features=layers,
            data_consistency=data_consistency,
        )

    def forward(
        self,
        x_kspace: torch.Tensor,
        x_kspace_mask: torch.Tensor,
        y_kspace: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
    ) -> torch.Tensor:
        output = {}
        for _ in range(self.steps):
            out = self.unet(
                x_kspace=x_kspace,
                x_kspace_mask=x_kspace_mask,
                y_kspace=y_kspace,
                name=name,
            )
            if "output" in out:
                x_kspace = out["output"]
            else:
                x_kspace = out

            output["output"] = x_kspace

            if "loss" in out:
                if "loss" in output:
                    output["loss"] += out["loss"]
                else:
                    output["loss"] = out["loss"]

        return output


if __name__ == "__main__":
    unet = UNet(in_channels=2, out_channels=2, features=[2, 2, 2, 2, 2])
    print(unet)
    input = torch.randn(1, 3, 256, 256)
    output = unet(input, input, input > 0)
    print(output.shape)

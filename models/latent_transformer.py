import math
import random
from typing import Optional, Sequence, Tuple, Union
from monai.networks.nets.basic_unet import BasicUNet as MonaiUNet
import torch
import torch.nn as nn

import torch
import torch.nn.functional as F
from torch import nn
import wandb

from models.utils import HardDataConsistency, image_to_kspace, kspace_to_image


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.0):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.0):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        out_channels: int,
        layers: Sequence[int] = (32, 64, 128, 256, 512),
        bilinear=True,
        predict_variance=False,
        dropout_rate=0.0,
        residual=False,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.layers = layers
        self.bilinear = bilinear
        self.predict_variance = predict_variance
        self.residual = residual

        self.inc = DoubleConv(2, layers[0], dropout_rate=dropout_rate)

        self.downs_data = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i, l in enumerate(layers):
            if i != len(layers) - 1:
                self.downs_data.append(Down(layers[i], layers[i + 1], dropout_rate=dropout_rate))

                up_channels = layers[len(layers) - i - 1] + layers[len(layers) - i - 2]

                self.ups.append(
                    Up(
                        up_channels,
                        layers[len(layers) - i - 2],
                        bilinear=bilinear,
                        dropout_rate=dropout_rate,
                    )
                )

        self.outc = OutConv(layers[0], self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_downs = self.down_forward(x)
        x_ups = self.up_forward(x_downs)

        logits = self.outc(x_ups[-1])

        if self.residual:
            logits = logits + x

        return logits

    def down_forward(self, x: torch.Tensor) -> torch.Tensor:
        x_downs = []

        x_downs.append(self.inc(x))

        for down in self.downs_data:
            x_downs.append(down(x_downs[-1]))

        return x_downs

    def up_forward(self, x_downs: Sequence[torch.Tensor]) -> torch.Tensor:
        x_ups = []

        for i, up in enumerate(self.ups):
            x_ups.append(up(x_downs[-1 - i], x_downs[-2 - i]))

        return x_ups


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 9):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.key_conv = nn.Conv2d(dimensions, dimensions, kernel_size=1)
        self.query_conv = nn.Conv2d(dimensions, dimensions, kernel_size=1)
        self.value_conv = nn.Conv2d(dimensions, dimensions, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys, queries, values = [], [], []
        for i in range(x.shape[1]):
            keys.append(self.key_conv(x[:, i]))
            queries.append(self.query_conv(x[:, i]))
            values.append(self.value_conv(x[:, i]))

        keys = torch.stack(keys, dim=1)
        queries = torch.stack(queries, dim=1)
        values = torch.stack(values, dim=1)

        outputs = []
        for i in range(x.shape[1]):
            weights = torch.softmax((keys[:, i : i + 1] * queries).sum(2), dim=1).unsqueeze(2)
            outputs.append((weights * values).sum(1))

        return torch.stack(outputs, dim=1) + x


class TemporalSelfAttention(nn.Module):
    def __init__(self, dims, heads):
        super().__init__()
        self.blocks = nn.ModuleList([TemporalSelfAttentionBlock(dims) for _ in range(heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for block in self.blocks:
            outputs.append(block(x))

        return torch.stack(outputs, dim=-1).mean(-1)


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, dims, heads, layers):
        super().__init__()
        self.layers = nn.ModuleList([TemporalSelfAttention(dims, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


class LatentTransformer(nn.Module):
    def __init__(
        self,
        out_channels: int,
        image_size: Tuple[int, int],
        layers: Sequence[int] = (32, 64, 128, 256, 512),
        bilinear=True,
        dropout_rate=0.0,
        residual=False,
        data_consistency: str = "none",
        transformer_mode: str = "encoder-decoder",  # "encoder-decoder" or "encoder-only"
    ) -> None:
        super().__init__()

        self.transformer_mode = transformer_mode
        conv_dims = layers[-1]

        self.unet_lt = UNet(
            out_channels=2,
            layers=layers,
            bilinear=bilinear,
            dropout_rate=dropout_rate,
            residual=residual,
        )

        self.unet_denoise = MonaiUNet(
            spatial_dims=2,
            in_channels=out_channels,
            out_channels=out_channels,
            features=(32, 32, 64, 128, 256, 32),
        )

        self.conv_in = nn.Conv2d(layers[-1], conv_dims, kernel_size=1)
        self.conv_out = nn.Conv2d(conv_dims, layers[-1], kernel_size=1)

        example_image = torch.zeros((1, 2, *image_size))
        downs = self.unet_lt.down_forward(example_image)
        down = self.conv_in(downs[-1][-1])
        down = down.flatten(0)
        embedding_dim = down.shape[0]

        self.positional_encoding = PositionalEncoding(embedding_dim, dropout_rate)

        if transformer_mode == "encoder-decoder":
            self.transformer = nn.Transformer(
                d_model=embedding_dim,
                nhead=1,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=64,
                dropout=dropout_rate,
                activation="relu",
                batch_first=True,
            )
        elif transformer_mode == "encoder-only":
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=4,
                    dim_feedforward=1024,
                    dropout=dropout_rate,
                    activation="relu",
                    batch_first=True,
                ),
                num_layers=2,
            )
        elif transformer_mode == "attention":
            self.attention = TemporalTransformerEncoder(conv_dims, 6, 4)
        elif transformer_mode == "image-encoder-only":
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=conv_dims,
                    nhead=8,
                    dim_feedforward=layers[-1],
                    dropout=dropout_rate,
                    activation="relu",
                    batch_first=True,
                ),
                num_layers=6,
            )
        elif transformer_mode == "image-encoder-decoder":
            self.transformer = nn.Transformer(
                d_model=conv_dims,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=layers[-1],
                dropout=dropout_rate,
                activation="relu",
                batch_first=True,
            )
        elif transformer_mode == "multi-scale-lt":
            self.test_positional_encodings = []
            self.test_attentions = []
            for layer in layers:
                self.test_positional_encodings.append(PositionalEncoding(layer, dropout_rate))
                self.test_attentions.append(TemporalTransformerEncoder(layer, 8, 6))

            self.test_positional_encodings = nn.ModuleList(self.test_positional_encodings)
            self.test_attentions = nn.ModuleList(self.test_attentions)

            self.conv_in = nn.Identity()
            self.conv_out = nn.Identity()
        elif transformer_mode != "none":
            raise ValueError(f"Unknown transformer mode {transformer_mode}")

        self.dc = None
        if data_consistency == "hard":
            self.dc = HardDataConsistency()
        elif data_consistency != "none":
            raise ValueError(f"Unknown data consistency method {data_consistency}")

    def forward(
        self,
        x_kspace: torch.Tensor,
        x_kspace_mask: torch.Tensor,
        y_kspace: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        norm: Optional[bool] = True,
    ) -> torch.Tensor:
        if norm:
            norm_factor = 1e4
        else:
            norm_factor = 1.0

        x_kspace = torch.abs(x_kspace) * norm_factor * torch.exp(1j * torch.angle(x_kspace))
        original_x_kspace = x_kspace.clone()

        return_dict = {"output": None, "loss": 0}

        # x_image = x_kspace  # temp
        (
            batch_size,
            timepoints,
            slices,
            height,
            width,
        ) = x_kspace.shape

        assert slices == 1, "Only single-slice data is supported"

        if y_kspace is not None:
            y_kspace = torch.abs(y_kspace) * norm_factor * torch.exp(1j * torch.angle(y_kspace))
            y_image = kspace_to_image(y_kspace)
            y_flat = torch.cat([torch.abs(y_image), torch.angle(y_image)], dim=1).squeeze(2)

        x_image = kspace_to_image(x_kspace)  # dim = (batch, time, 1, height, width)

        denoised_image = self.unet_denoise(torch.cat([torch.abs(x_image), torch.angle(x_image)], dim=1).squeeze(2))
        denoised_image = denoised_image[:, :timepoints] * torch.exp(1j * denoised_image[:, timepoints:])
        denoised_image = denoised_image.unsqueeze(2) + x_image
        denoised_kspace = image_to_kspace(denoised_image)

        if self.transformer_mode != "none":
            outputs = []
            for i in range(timepoints):
                x = x_image[:, i : (i + 1)]
                x = torch.cat([torch.abs(x), torch.angle(x)], dim=1).squeeze(2)

                outputs.append(self.unet_lt.down_forward(x))

            for i, output_layer in enumerate(outputs[0]):
                final_outputs = [out[i] for out in outputs]
                conved_final_outputs = [self.conv_in(out) for out in final_outputs]
                conved_final_outputs = torch.stack(conved_final_outputs, dim=1)
                latent_codes = conved_final_outputs.view(batch_size, timepoints, -1)

                if self.transformer_mode == "encoder-decoder":
                    latent_codes = self.transformer(latent_codes, latent_codes)
                elif self.transformer_mode == "encoder-only":
                    latent_codes = self.transformer_encoder(latent_codes)
                elif self.transformer_mode == "attention":
                    latent_codes = self.attention(conved_final_outputs)
                elif self.transformer_mode == "image-encoder-only":
                    reshaped_conved_final_outputs = conved_final_outputs.permute(0, 3, 4, 1, 2)
                    flattened_conved_final_outputs = reshaped_conved_final_outputs.flatten(0, 2)
                    latent_codes = self.transformer_encoder(flattened_conved_final_outputs)
                    latent_codes = latent_codes.view(*reshaped_conved_final_outputs.shape)
                    latent_codes = latent_codes.permute(0, 3, 4, 1, 2)
                elif self.transformer_mode == "image-encoder-decoder":
                    reshaped_conved_final_outputs = conved_final_outputs.permute(0, 3, 4, 1, 2)
                    flattened_conved_final_outputs = reshaped_conved_final_outputs.flatten(0, 2)
                    latent_codes = self.transformer(flattened_conved_final_outputs, flattened_conved_final_outputs)
                    latent_codes = latent_codes.view(*reshaped_conved_final_outputs.shape)
                    latent_codes = latent_codes.permute(0, 3, 4, 1, 2)
                elif self.transformer_mode == "multi-scale-lt":
                    reshaped_conved_final_outputs = conved_final_outputs.permute(0, 3, 4, 1, 2)
                    flattened_conved_final_outputs = reshaped_conved_final_outputs.flatten(0, 2)

                    latent_codes = self.test_positional_encodings[i](flattened_conved_final_outputs)

                    latent_codes = latent_codes.view(*reshaped_conved_final_outputs.shape)
                    latent_codes = latent_codes.permute(0, 3, 4, 1, 2)

                    latent_codes = self.test_attentions[i](latent_codes)

                latent_codes = latent_codes.view(*conved_final_outputs.shape)
                latent_codes = [self.conv_out(latent_codes[:, i : (i + 1)].squeeze(1)) for i in range(timepoints)]

                # residual learning for latent transformer
                latent_codes = [latent_codes[i] + final_outputs[i] for i in range(timepoints)]

                for j in range(timepoints):
                    outputs[j][i] = latent_codes[j]

            # --- latent transformer ---

            new_outputs = []
            for i in range(timepoints):
                out = self.unet_lt.up_forward(outputs[i])
                out = self.unet_lt.outc(out[-1])
                new_outputs.append(out)

            channels = new_outputs[0].shape[1] // 2
            out = torch.cat([no[:, :channels] for no in new_outputs] + [no[:, channels:] for no in new_outputs], dim=1)

            out = out[:, :timepoints] * torch.exp(1j * out[:, timepoints:])
            out = out.unsqueeze(2) + denoised_image
        else:
            out = denoised_image

        if y_kspace is not None:
            return_dict["loss"] = return_dict["loss"] + (out - y_image).abs().mean()

        out = image_to_kspace(out)

        if self.dc is not None:
            out = self.dc(out, original_x_kspace, x_kspace_mask[:, None, None].repeat(1, timepoints, 1, 1, 1))

        out = out.abs() / norm_factor * torch.exp(1j * out.angle())
        return_dict["output"] = out
        return return_dict


if __name__ == "__main__":
    example = torch.randn(2, 9, 1, 256, 256, dtype=torch.complex64)
    mask = torch.zeros(2, 256, 256)
    target = torch.randn_like(example)

    lt = LatentTransformer(
        2,
        image_size=(256, 256),
        layers=(2, 3, 4, 5),
        residual=True,
        transformer_mode="encoder-only",
        data_consistency="hard",
    )

    out = lt(example, mask, target, name="t1")

    print(out["output"].shape)

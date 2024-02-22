from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.latent_transformer import PositionalEncoding, TemporalTransformerEncoder, UNet

from models.utils import HardDataConsistency, image_to_kspace, kspace_to_image


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(
            hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, groups=hidden_channels * 2, bias=False
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(
            self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
            .transpose(-2, -1)
            .contiguous()
            .reshape(b, c, h, w)
        )
        x = x + self.ffn(
            self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
            .transpose(-2, -1)
            .contiguous()
            .reshape(b, c, h, w)
        )
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False), nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False), nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class RestormerCore(nn.Module):
    def __init__(
        self,
        num_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        channels=[48, 96, 192, 384],
        num_refinement=4,
        expansion_factor=2.66,
        in_out_channels=3,
    ):
        super(RestormerCore, self).__init__()
        self.embed_conv = nn.Conv2d(in_out_channels, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(*[TransformerBlock(num_ch, num_ah, expansion_factor) for _ in range(num_tb)])
                for num_tb, num_ah, num_ch in zip(num_blocks, num_heads, channels)
            ]
        )
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                for i in reversed(range(2, len(channels)))
            ]
        )
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[TransformerBlock(channels[2], num_heads[2], expansion_factor) for _ in range(num_blocks[2])]
                )
            ]
        )
        self.decoders.append(
            nn.Sequential(
                *[TransformerBlock(channels[1], num_heads[1], expansion_factor) for _ in range(num_blocks[1])]
            )
        )
        # the channel of last one is not change
        self.decoders.append(
            nn.Sequential(
                *[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_blocks[0])]
            )
        )

        self.refinement = nn.Sequential(
            *[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_refinement)]
        )
        self.output = nn.Conv2d(channels[1], in_out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr) + x

        return out

    def down_forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        return [out_enc1, out_enc2, out_enc3, out_enc4]

    def up_forward(self, out_enc, x=None, residual=True):
        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc[3]), out_enc[2]], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc[1]], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc[0]], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr)

        if residual and x is not None:
            out = out + x

        return out


class Restormer(nn.Module):
    def __init__(
        self,
        num_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        channels=[48, 96, 192, 384],
        num_refinement=4,
        expansion_factor=2.66,
        in_out_channels=3,
        data_consistency="none",
        steps: int = 1,
        intermediate_image_losses: bool = False,
        latent_transformer_mode: str = "none",
        transformer_conv_dims: int = 64,
        transformer_ff_dims: int = 256,
        transformer_n_heads: int = 4,
        transformer_n_layers: int = 4,
    ):
        super(Restormer, self).__init__()
        self.steps = steps
        self.intermediate_image_losses = intermediate_image_losses
        self.latent_transformer_mode = latent_transformer_mode

        self.restormer = RestormerCore(
            num_blocks, num_heads, channels, num_refinement, expansion_factor, in_out_channels
        )

        if latent_transformer_mode != "none":
            lt_layers = [max(c // 2, 2 ** (i + 1)) for i, c in enumerate(channels)]
            self.restormer_lt = RestormerCore(num_blocks, num_heads, lt_layers, num_refinement, expansion_factor, 2)

        self.dc = None
        if data_consistency == "hard":
            self.dc = HardDataConsistency()
        elif data_consistency != "none":
            raise ValueError(f"Unknown data consistency: {data_consistency}")

        self.conv_in = nn.Conv2d(channels[-1], transformer_conv_dims, kernel_size=1, bias=False)
        self.conv_out = nn.Conv2d(transformer_conv_dims, channels[-1], kernel_size=1, bias=False)

        self.latent_transformer = None
        if self.latent_transformer_mode == "encoder-decoder":
            self.transformer = nn.Transformer(
                d_model=18 * 64 * transformer_conv_dims,
                nhead=transformer_n_heads,
                num_encoder_layers=transformer_n_layers,
                num_decoder_layers=transformer_n_layers,
                dim_feedforward=transformer_ff_dims,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            )
        elif self.latent_transformer_mode == "encoder-only":
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=18 * 64 * transformer_conv_dims,
                    nhead=transformer_n_heads,
                    dim_feedforward=transformer_ff_dims,
                    dropout=0.1,
                    activation="relu",
                    batch_first=True,
                ),
                num_layers=transformer_n_layers,
            )
        elif self.latent_transformer_mode == "lstm":
            self.transformer = nn.LSTM(
                input_size=18 * 64 * transformer_conv_dims,
                hidden_size=transformer_ff_dims,
                num_layers=transformer_n_layers,
                batch_first=True,
                bidirectional=True,
            )
        elif self.latent_transformer_mode == "attention":
            self.attention = TemporalTransformerEncoder(transformer_conv_dims, 6, 4)
        elif self.latent_transformer_mode == "image-encoder-only":
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=transformer_conv_dims,
                    nhead=8,
                    dim_feedforward=channels[-1],
                    dropout=0.1,
                    activation="relu",
                    batch_first=True,
                ),
                num_layers=6,
            )
        elif self.latent_transformer_mode == "image-encoder-decoder":
            self.transformer = nn.Transformer(
                d_model=transformer_conv_dims,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=channels[-1],
                dropout=0.1,
                activation="relu",
                batch_first=True,
            )
        elif self.latent_transformer_mode == "none":
            self.conv_in = nn.Identity()
            self.conv_out = nn.Identity()
        elif self.latent_transformer_mode == "multi-scale-lt":
            self.positional_encodings = []
            self.attentions = []
            for layer in lt_layers:
                self.positional_encodings.append(PositionalEncoding(layer, 0.0))
                self.attentions.append(TemporalTransformerEncoder(layer, transformer_n_heads, transformer_n_layers))

            self.positional_encodings = nn.ModuleList(self.positional_encodings)
            self.attentions = nn.ModuleList(self.attentions)

            self.conv_in = nn.Identity()
            self.conv_out = nn.Identity()
        else:
            raise ValueError(f"Unknown transformer mode {self.latent_transformer_mode}")

    def forward(
        self,
        x_kspace: torch.Tensor,
        x_kspace_mask: torch.Tensor,
        y_kspace: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        norm: Optional[bool] = True,
    ):
        if norm:
            norm_factor = 1e2
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

        for step in range(self.steps):
            x_image = kspace_to_image(x_kspace)  # dim = (batch, time, 1, height, width)

            # ---- use the restormer to de-noise the images ----
            cat_image = torch.cat([torch.abs(x_image), torch.angle(x_image)], dim=1).squeeze(2)
            if "t1" in name.lower():
                cat_image[:, :3] = -cat_image[:, :3]

            # check that all the dimensions are divisible by 8 and pad them if not
            original_shape = cat_image.shape[-2:]
            if cat_image.shape[-1] % 8 != 0:
                cat_image = F.pad(cat_image, (0, 8 - cat_image.shape[-1] % 8))

            if cat_image.shape[-2] % 8 != 0:
                cat_image = F.pad(cat_image, (0, 0, 0, 8 - cat_image.shape[-2] % 8))

            out = self.restormer(cat_image)

            # unpad the output
            out = out[..., : original_shape[0], : original_shape[1]]

            if "t1" in name.lower():
                out[:, :3] = -out[:, :3]

            denoised_image = out
            denoised_image = denoised_image[:, :timepoints] * torch.exp(1j * denoised_image[:, timepoints:])
            denoised_image = denoised_image.unsqueeze(2)
            denoised_kspace = image_to_kspace(denoised_image)
            # ---- end of restormer denoising ----

            # ---- latent transformer ----
            if self.latent_transformer_mode != "none":
                outputs = []
                for i in range(timepoints):
                    x = x_image[:, i : (i + 1)]
                    x = torch.cat([torch.abs(x), torch.angle(x)], dim=1).squeeze(2)

                    if x.shape[-1] % 8 != 0:
                        x = F.pad(x, (0, 8 - x.shape[-1] % 8))

                    if x.shape[-2] % 8 != 0:
                        x = F.pad(x, (0, 0, 0, 8 - x.shape[-2] % 8))

                    outputs.append(self.restormer_lt.down_forward(x))

                for i, output_layer in enumerate(outputs[0]):
                    final_outputs = [out[i] for out in outputs]
                    conved_final_outputs = [self.conv_in(out) for out in final_outputs]
                    conved_final_outputs = torch.stack(conved_final_outputs, dim=1)
                    latent_codes = conved_final_outputs.view(batch_size, timepoints, -1)

                    # ---- per-layer latent transformer ----
                    if self.latent_transformer_mode == "encoder-decoder":
                        latent_codes = self.transformer(latent_codes, latent_codes)
                    elif self.latent_transformer_mode == "encoder-only":
                        latent_codes = self.transformer_encoder(latent_codes)
                    elif self.latent_transformer_mode == "attention":
                        latent_codes = self.attention(conved_final_outputs)
                    elif self.latent_transformer_mode == "image-encoder-only":
                        reshaped_conved_final_outputs = conved_final_outputs.permute(0, 3, 4, 1, 2)
                        flattened_conved_final_outputs = reshaped_conved_final_outputs.flatten(0, 2)
                        latent_codes = self.transformer_encoder(flattened_conved_final_outputs)
                        latent_codes = latent_codes.view(*reshaped_conved_final_outputs.shape)
                        latent_codes = latent_codes.permute(0, 3, 4, 1, 2)
                    elif self.latent_transformer_mode == "image-encoder-decoder":
                        reshaped_conved_final_outputs = conved_final_outputs.permute(0, 3, 4, 1, 2)
                        flattened_conved_final_outputs = reshaped_conved_final_outputs.flatten(0, 2)
                        latent_codes = self.transformer(flattened_conved_final_outputs, flattened_conved_final_outputs)
                        latent_codes = latent_codes.view(*reshaped_conved_final_outputs.shape)
                        latent_codes = latent_codes.permute(0, 3, 4, 1, 2)
                    elif self.latent_transformer_mode == "multi-scale-lt":
                        reshaped_conved_final_outputs = conved_final_outputs.permute(0, 3, 4, 1, 2)
                        flattened_conved_final_outputs = reshaped_conved_final_outputs.flatten(0, 2)

                        latent_codes = self.positional_encodings[i](flattened_conved_final_outputs)

                        latent_codes = latent_codes.view(*reshaped_conved_final_outputs.shape)
                        latent_codes = latent_codes.permute(0, 3, 4, 1, 2)

                        latent_codes = self.attentions[i](latent_codes)

                    latent_codes = latent_codes.view(*conved_final_outputs.shape)
                    latent_codes = [self.conv_out(latent_codes[:, i : (i + 1)].squeeze(1)) for i in range(timepoints)]

                    # residual learning for latent transformer
                    latent_codes = [latent_codes[i] + final_outputs[i] for i in range(timepoints)]

                    for j in range(timepoints):
                        outputs[j][i] = latent_codes[j]

                # --- end of per-layer latent transformer ---

                new_outputs = []
                for i in range(timepoints):
                    out = self.restormer_lt.up_forward(outputs[i])
                    new_outputs.append(out)

                channels = new_outputs[0].shape[1] // 2
                out = torch.cat(
                    [no[:, :channels] for no in new_outputs] + [no[:, channels:] for no in new_outputs], dim=1
                )

                # crop to the original size
                out = out[..., : original_shape[0], : original_shape[1]]

                out = out[:, :timepoints] * torch.exp(1j * out[:, timepoints:])
                out = out.unsqueeze(2) + denoised_image
            else:
                out = denoised_image
            # ---- end of latent transformer ----

            out = image_to_kspace(out)

            if self.dc is not None:
                out = self.dc(out, original_x_kspace, x_kspace_mask[:, None, None].repeat(1, timepoints, 1, 1, 1))

            if (self.intermediate_image_losses or step == self.steps - 1) and y_kspace is not None:
                return_dict["loss"] += (out - y_kspace).abs().mean()

            x_kspace = out

        out = out.abs() / norm_factor * torch.exp(1j * out.angle())
        return_dict["output"] = out
        return return_dict

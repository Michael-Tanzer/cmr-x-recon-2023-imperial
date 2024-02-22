from typing import Optional
import torch
import torch.nn as nn
from monai.networks.nets.basic_unet import BasicUNet as UNet
from mapping import compute_t2_mapping

from models.utils import image_to_kspace, kspace_to_image


class ParametricT2Predict(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(
            spatial_dims=2,
            in_channels=6,
            out_channels=2,
            features=[32, 32, 64, 128, 256, 32],
        )

    def forward(
        self,
        x_kspace: torch.Tensor,
        x_kspace_mask: torch.Tensor,
        y_kspace: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
    ) -> torch.Tensor:
        B = x_kspace.shape[0]
        times = torch.zeros(B, 3, 1).to(x_kspace.device)
        times[:] = torch.tensor([0, 35, 55]).view(3, 1)

        x_kspace = torch.abs(x_kspace) * torch.exp(1j * torch.angle(x_kspace))
        original_x_kspace = x_kspace.clone()

        x_images = kspace_to_image(x_kspace)
        cat_images = torch.cat([torch.abs(x_images), torch.angle(x_images)], dim=1).squeeze(2)

        output_model = self.unet(cat_images)

        predicted_log_images = []
        for i in range(3):
            predicted_log_images.append(
                output_model[:, 0, :, :] + times[:, i, 0].view(B, 1, 1) * output_model[:, 1, :, :]
            )

        predicted_log_images = (torch.stack(predicted_log_images, dim=1) + torch.log(cat_images[:, :3] + 1)).unsqueeze(
            2
        )
        predicted_images = torch.exp(predicted_log_images) - 1
        predicted_t2 = -1 / (output_model[:, 1])

        loss = None
        if y_kspace is not None:
            t2_target = compute_t2_mapping(
                times.cpu().detach().numpy().squeeze(2),
                y_kspace.cpu().detach().numpy().swapaxes(0, 2).squeeze(0),
            )
            t2_target = torch.from_numpy(t2_target).to(x_kspace.device)
            t2_target = torch.flip(t2_target, dims=(-1, -2))
            y_images = kspace_to_image(y_kspace).abs()
            y_log_images = torch.log(y_images + 1)
            t2_loss_mask = (y_images > torch.quantile(y_images, 0.75)).squeeze(2).max(dim=1).values

            loss_images = torch.abs(y_log_images - predicted_log_images).mean()
            loss_t2 = torch.abs(t2_target - predicted_t2)
            loss_t2 = loss_t2[t2_loss_mask].mean()

            loss = loss_images

        predicted_kspaces = predicted_images * torch.exp(1j * torch.angle(original_x_kspace))
        predicted_kspaces = image_to_kspace(predicted_kspaces)

        return {
            "loss": loss,
            "output": predicted_kspaces,
        }

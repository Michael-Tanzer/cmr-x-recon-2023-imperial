import argparse
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from functools import partial
import logging
import math
import os
import time
from typing import Literal, Union
import numpy as np
import pyrootutils
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
from dataloading import build_dataloaders
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    mean_squared_error,
)
from ensemble.base import EnsembleOptimiserWrapper, EnsembleSchedulerWrapper
from ensemble.factory import EnsembleFactory
from mapping import compute_t1_mapping, compute_t2_mapping
from models.utils import image_to_kspace, kspace_to_image, combine_coil_img

from utils import (
    compare_configs,
    get_loss_function_from_name,
    get_model_from_name,
    log_image_wandb,
    seed_everything,
    set_custom_repr_tensors,
)
from config import parser


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def train(
    model: nn.Module,
    dataloader_train: torch.utils.data.DataLoader,
    dataloader_valid: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    logger: wandb.wandb_sdk.wandb_run.Run,
    config: Namespace,
):
    model.train()
    step = 0

    if not config.debug:
        savepath = create_save_dir(config)

    for epoch in range(config.epochs):
        model.train()
        for batch_id, (x, y) in tqdm(
            enumerate(dataloader_train),
            total=len(dataloader_train),
            desc=f"Epoch {epoch}",
        ):
            x = x.to(config.device)
            y = y.to(config.device)

            optimizer.zero_grad()

            x_kspaces = []
            x_kspace_masks = []
            y_kspaces = []

            if "t1" in config.tasks:
                x_kspaces.append(x.t1_kspace)
                x_kspace_masks.append(x.t1_kspace_mask)
                y_kspaces.append(y.t1_kspace)

            if "t2" in config.tasks:
                x_kspaces.append(x.t2_kspace)
                x_kspace_masks.append(x.t2_kspace_mask)
                y_kspaces.append(y.t2_kspace)

            loss = None
            for name, x_kspace, x_kspace_mask, y_kspace in zip(config.tasks, x_kspaces, x_kspace_masks, y_kspaces):
                with autocast(enabled=config.mixed_precision):
                    output = model(x_kspace, x_kspace_mask, y_kspace, name)

                    if loss is None:
                        loss = output["loss"]
                    else:
                        loss = output["loss"] + loss

                    loss = loss_fn(output["output"], getattr(y, f"{name}_kspace")) + loss

                if batch_id == len(dataloader_train) - 1 and epoch % config.log_freq == 0:
                    if config.coils == "multi":
                        # the output is the same dim as input of multi-coil kpace.
                        image_output = combine_coil_img(kspace_to_image(output["output"]), axis=-3).detach().cpu()
                        image_target = combine_coil_img(kspace_to_image(y_kspace), axis=-3).detach().cpu()
                        lamdba = output["lam"].detach().cpu()
                        csm = output["csm"].detach().cpu()
                        logger.log({"train/lambda": lamdba.item(), "step": step})
                        # logg the csm to see the update of csm during training
                        log_image_wandb([csm[0].abs()], f"train/csm {name.upper()}", step)

                    else:
                        image_output = kspace_to_image(output["output"].detach().cpu()).abs()
                        image_target = kspace_to_image(y_kspace.detach().cpu()).abs()
                    image_output[image_output > image_target.max()] = image_target.max()
                    image_output[image_output < 0] = 0

                    log_image_wandb(
                        [image_output[0, :, 0], image_target[0, :, 0]], f"train/images {name.upper()}", step
                    )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()

            if batch_id % config.log_freq == 0:
                logger.log({"train/loss": loss.item(), "step": step, "train/lr": lr_scheduler.get_last_lr()[0]})

            step += 1

        dataloader_train.dataset.set_next_indices(
            [i for batch in dataloader_train.batch_sampler.batches for i in batch]
        )

        if epoch % config.val_freq == 0:
            # try to combine th val_freq and log_freq together
            val_metrics = validation(dataloader_valid, model, loss_fn, config.device)
            logger.log(val_metrics | {"step": step})
            if not config.debug:
                saved_num = 2  # here just save the best 2 models based on the validation set.
                save(model, optimizer, config, val_metrics, savepath, saved_num, step)

        if not epoch % (config.epochs // config.number_ensemble_models) and epoch != 0:
            print("====== Starting next model ======")
            model.start_next_model()
            optimizer.start_next_model()
            lr_scheduler.start_next_model()

    if config.epochs > 0:
        save(model, optimizer, config, val_metrics, savepath, saved_num, step)

    return model


@torch.no_grad()
def validation(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: Union[torch.device, str],
    task_name: Literal["val", "test"] = "val",
    save_output: str = None,
):
    model.eval()

    losses = []
    outputs = []
    targets = []

    metrics = {}

    if "t1" in config.tasks:
        metrics |= {
            "T1_PSNR": [],
            "T1_SSIM": [],
            "T1_RMSE": [],
            "T1_NMSE": [],
            "T1_PSNR_MAP": [],
            "T1_SSIM_MAP": [],
            "T1_RMSE_MAP": [],
            "T1_NMSE_MAP": [],
        }

    if "t2" in config.tasks:
        metrics |= {
            "T2_PSNR": [],
            "T2_SSIM": [],
            "T2_RMSE": [],
            "T2_NMSE": [],
            "T2_PSNR_MAP": [],
            "T2_SSIM_MAP": [],
            "T2_RMSE_MAP": [],
            "T2_NMSE_MAP": [],
        }

    for ii, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        x_kspaces = []
        x_kspace_masks = []
        y_kspaces = []

        if "t1" in config.tasks:
            x_kspaces.append(x.t1_kspace)
            x_kspace_masks.append(x.t1_kspace_mask)
            y_kspaces.append(y.t1_kspace)

        if "t2" in config.tasks:
            x_kspaces.append(x.t2_kspace)
            x_kspace_masks.append(x.t2_kspace_mask)
            y_kspaces.append(y.t2_kspace)

        y_pred = {}
        loss = None
        for name, x_kspace, x_kspace_mask, y_kspace in zip(config.tasks, x_kspaces, x_kspace_masks, y_kspaces):
            output = model(x_kspace, x_kspace_mask, y_kspace, name)
            y_pred[f"{name}_output"] = output["output"]

            if loss is None:
                loss = output["loss"]
            else:
                loss += output["loss"]

            loss += loss_fn(output["output"], getattr(y, f"{name}_kspace"))

        if isinstance(loss, torch.Tensor) and loss.device != "cpu":
            loss = loss.cpu()

        losses.extend(loss for _ in range(x.t1_kspace.shape[0]))
        outputs.append({k: v.cpu() for k, v in y_pred.items()})
        targets.append(y.to("cpu"))

        # log images and image-based metrics
        for name in config.tasks:
            if config.coils == "multi":
                batch, timepoints, slices, ncoil, height, width = y_pred[f"{name}_output"].shape
                targets_kspace = y.t1_kspace if name == "t1" else y.t2_kspace
                # the output of the modl is coil_combined image and the target is coil_separate kspace
                targets_combined_image = combine_coil_img(kspace_to_image(targets_kspace))
                predictions_combined_image = combine_coil_img(kspace_to_image(y_pred[f"{name}_output"]))

                targets_images = targets_combined_image.reshape(batch * timepoints * slices, height, width).cpu()
                predictions_images = predictions_combined_image.reshape(
                    batch * timepoints * slices, height, width
                ).cpu()
            else:
                batch, timepoints, slices, height, width = y_pred[f"{name}_output"].shape
                targets_kspace = y.t1_kspace if name == "t1" else y.t2_kspace
                flat_predictions = y_pred[f"{name}_output"].reshape(batch * timepoints * slices, height, width)
                flat_targets = targets_kspace.reshape(batch * timepoints * slices, height, width)
                predictions_images = kspace_to_image(flat_predictions.cpu()).abs()
                targets_images = kspace_to_image(flat_targets.cpu()).abs()

            predictions_images = predictions_images[
                :,
                height // 2 - height // 4 : height // 2 + height // 4,
                width // 2 - width // 6 : width // 2 + width // 6,
            ]
            targets_images = targets_images[
                :,
                height // 2 - height // 4 : height // 2 + height // 4,
                width // 2 - width // 6 : width // 2 + width // 6,
            ]
            height, width = predictions_images.shape[-2:]

            for i in range(batch * timepoints * slices):
                metrics[f"{name.upper()}_PSNR"].append(
                    peak_signal_noise_ratio(predictions_images[i], targets_images[i])
                )
                metrics[f"{name.upper()}_SSIM"].append(
                    structural_similarity_index_measure(
                        predictions_images[i, None, None],
                        targets_images[i, None, None],
                    )
                )
                metrics[f"{name.upper()}_RMSE"].append(
                    mean_squared_error(predictions_images[i], targets_images[i]).sqrt()
                )
                metrics[f"{name.upper()}_NMSE"].append(
                    torch.norm(targets_images[i] - predictions_images[i]) ** 2 / torch.norm(targets_images[i]) ** 2
                )

            if ii == 0:
                predictions_images = predictions_images.reshape(batch, timepoints, slices, height, width)
                targets_images = targets_images.reshape(batch, timepoints, slices, height, width)
                predictions_images[predictions_images > targets_images.max()] = targets_images.max()
                predictions_images[predictions_images < 0] = 0

                log_image_wandb(
                    [predictions_images[0, :, 0], targets_images[0, :, 0]],
                    f"{task_name}/Images {name.upper()}",
                    step=step,
                )

    dataloader.dataset.set_next_indices([i for batch in dataloader.batch_sampler.batches for i in batch])
    model.train()
    if config.single_slice:
        batched_outputs = []
        batched_targets = []

        # flatten the outputs
        new_outputs = defaultdict(lambda: None)
        for k, v in outputs[0].items():
            if "output" not in k:
                continue

            for output in outputs:
                if new_outputs[k] is None:
                    new_outputs[k] = output[k]
                else:
                    new_outputs[k] = torch.cat([new_outputs[k], output[k]], dim=0)

        outputs = new_outputs

        # flatten the targets
        new_targets = None
        for target in targets:
            if new_targets is None:
                new_targets = deepcopy(target)
            else:
                for k, v in target.__dict__.items():
                    if isinstance(v, torch.Tensor):
                        new_targets.__dict__[k] = torch.cat([new_targets.__dict__[k], v], dim=0)
                    elif v is None:
                        new_targets.__dict__[k] = None
                    elif isinstance(v, bool) or isinstance(v, int) or isinstance(v, float):
                        new_targets.__dict__[k] = v
                    else:
                        new_targets.__dict__[k].extend(v)
        targets = new_targets

        i = 0
        while (outputs[f"{name}_output"].shape[0] - i) > 0:
            number_slices = targets.original_number_slices[0].item()
            batched_targets.append(targets[:number_slices])
            targets = targets[number_slices:]
            new_batched_outputs = {}
            for k, v in outputs.items():
                new_batched_outputs[k] = v[i : i + number_slices]
            batched_outputs.append(new_batched_outputs)
            i += number_slices

        outputs = batched_outputs
        targets = batched_targets

        for i in range(len(outputs)):
            for name in config.tasks:
                # get the times depending on the name
                times = (
                    torch.stack([getattr(targets[i], f"{name}_times")[j] for j in range(len(targets[i]))])
                    .squeeze()
                    .numpy()
                )
                target_kspace = torch.stack(
                    [getattr(targets[i], f"{name}_kspace")[j] for j in range(len(targets[i]))]
                ).squeeze()

                output_kspace = outputs[i][f"{name}_output"].squeeze()

                if times.ndim == 1:
                    times = times[None, :]
                    target_kspace = target_kspace[None, :]
                    output_kspace = output_kspace[None, :]

                if config.coils == "multi":
                    slices, timesteps, coils, height, width = target_kspace.shape
                    target_kspace = image_to_kspace(
                        kspace_to_image(target_kspace)[
                            ...,
                            height // 2 - height // 4 : height // 2 + height // 4,
                            width // 2 - width // 6 : width // 2 + width // 6,
                        ]
                    ).numpy()
                    output_kspace = image_to_kspace(
                        kspace_to_image(output_kspace)[
                            ...,
                            height // 2 - height // 4 : height // 2 + height // 4,
                            width // 2 - width // 6 : width // 2 + width // 6,
                        ]
                    ).numpy()
                else:
                    slices, timesteps, height, width = target_kspace.shape
                    target_kspace = image_to_kspace(
                        kspace_to_image(target_kspace)[
                            ...,
                            height // 2 - height // 4 : height // 2 + height // 4,
                            width // 2 - width // 6 : width // 2 + width // 6,
                        ]
                    ).numpy()
                    output_kspace = image_to_kspace(
                        kspace_to_image(output_kspace)[
                            ...,
                            height // 2 - height // 4 : height // 2 + height // 4,
                            width // 2 - width // 6 : width // 2 + width // 6,
                        ]
                    ).numpy()

                # compute the mapping
                if name == "t1":
                    target = torch.from_numpy(
                        compute_t1_mapping(times, target_kspace, multi_coil=config.coils == "multi")
                    )
                    output = torch.from_numpy(
                        compute_t1_mapping(times, output_kspace, multi_coil=config.coils == "multi")
                    )
                else:
                    target = torch.from_numpy(
                        compute_t2_mapping(times, target_kspace, multi_coil=config.coils == "multi")
                    )
                    output = torch.from_numpy(
                        compute_t2_mapping(times, output_kspace, multi_coil=config.coils == "multi")
                    )

                for s in range(target.shape[0]):
                    metrics[f"{name.upper()}_PSNR_MAP"].append(peak_signal_noise_ratio(output[s], target[s]))
                    metrics[f"{name.upper()}_SSIM_MAP"].append(
                        structural_similarity_index_measure(output[s][None, None], target[s][None, None])
                    )
                    metrics[f"{name.upper()}_RMSE_MAP"].append(mean_squared_error(output[s], target[s]).sqrt())
                    metrics[f"{name.upper()}_NMSE_MAP"].append(
                        torch.norm(target[s] - output[s]) ** 2 / torch.norm(target[s]) ** 2
                    )

                if i == 0:
                    output[output > target.max()] = target.max()
                    output[output < 0] = 0
                    log_image_wandb([output, target], f"{task_name}/Images {name.upper()} mapping", step)

    else:
        raise NotImplementedError("Not implemented for non-single slice")
        for name in config.tasks:
            t1_times = [targets[i].t1_times for i in range(len(targets))]
            t2_times = [targets[i].t2_times for i in range(len(targets))]
            t1_target_kspace = [targets[i].t1_kspace for i in range(len(targets))]
            t2_target_kspace = [targets[i].t2_kspace for i in range(len(targets))]
            t1_output_kspace = [outputs[i]["t1_output"] for i in range(len(outputs))]
            t2_output_kspace = [outputs[i]["t2_output"] for i in range(len(outputs))]

            for i in range(len(t1_times)):
                target_t1 = compute_t1_mapping(t1_times[i], t1_target_kspace[i])
                output_t1 = compute_t1_mapping(t1_times[i], t1_output_kspace[i])

                target_t2 = compute_t2_mapping(t2_times[i], t2_target_kspace[i])
                output_t2 = compute_t2_mapping(t2_times[i], t2_output_kspace[i])

                metrics["T1_PSNR"].append(peak_signal_noise_ratio(output_t1, target_t1))
                metrics["T1_SSIM"].append(
                    structural_similarity_index_measure(output_t1[None, None], target_t1[None, None])
                )
                metrics["T1_RMSE"].append(mean_squared_error(output_t1, target_t1).sqrt())

                metrics["T2_PSNR"].append(peak_signal_noise_ratio(output_t2, target_t2))
                metrics["T2_SSIM"].append(
                    structural_similarity_index_measure(output_t2[None, None], target_t2[None, None])
                )
                metrics["T2_RMSE"].append(mean_squared_error(output_t2, target_t2).sqrt())

                log_image_wandb([output_t1, target_t1], f"{task_name}/T1 sample images", step)
                log_image_wandb([output_t2, target_t2], f"{task_name}/T2 sample images", step)

    return_metrics = {}
    return_metrics |= {f"{task_name}/loss": np.mean(losses)}
    return_metrics |= {f"{task_name}/{k}:": np.mean([vi for vi in v if torch.isfinite(vi)]) for k, v in metrics.items()}
    model.train()
    return return_metrics


def create_save_dir(
    config: Namespace,
):
    """
    similar to the training dataset, the save path is created using the config
    logdir/coils/tasks/AF/model/latent_transformer_mode
    please add more config if needed
    """
    savesubpath = [
        config.log_dir,
        config.coils,
        config.tasks[0],
        "AF_" + str(config.acc_factor),
        config.model,
        config.latent_transformer_mode,
    ]
    # for modl, add the corresponding denoising model
    if config.model == "modl":
        savesubpath.append(config.modl_denoising_model)
    savesubpath = [x for x in savesubpath if x is not None]
    # create the subpath using the config
    savepath = "/".join(savesubpath)
    # if subpath not exist, create it
    os.makedirs(savepath, exist_ok=True)
    return savepath


def save(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Namespace,
    val_metrics: dict,
    path: str,
    saved_num: int,
    step: int,
):
    # get the number of pt under the dir:
    files = os.listdir(path)
    losses, filename = [], []
    loss = val_metrics["val/loss"]
    savedname = "model_step{:04d}_loss{:.8f}.pt".format(step, loss)
    if len(files) < saved_num:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "step": step,
            },
            os.path.join(path, savedname),
        )
    else:
        # get the losses from the existing files, if the new loss is lower, save the lowest 2
        for file in files:
            loss = file.split("_loss")[-1].split(".pt")[0]
            losses.append(float(loss))
            filename.append(file)

        losses.append(val_metrics["val/loss"])
        # arrange the losses in ascending order and get the index
        sorted_indices = [i for i, _ in sorted(enumerate(losses), key=lambda x: x[1])]

        if sorted_indices[saved_num] != saved_num:
            # delete the file with the highest loss
            os.remove(os.path.join(path, filename[sorted_indices[saved_num]]))
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "step": step,
                },
                os.path.join(path, savedname),
            )


def load(new_config: Namespace, path):
    checkpoint = torch.load(path, map_location=torch.device(new_config.device))
    old_config = checkpoint["config"]
    compare_configs(old_config, new_config)
    model = get_model_from_name(old_config.model, old_config)
    step = checkpoint["step"]
    return model, step


if __name__ == "__main__":
    set_custom_repr_tensors()

    parser = argparse.ArgumentParser(parents=[parser], add_help=False)
    config = parser.parse_args()

    # Random seed
    seed_everything(config.seed)

    # Logger
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    logger = wandb.init(project="CMRxRecon2023", config=config, resume="allow", entity="cmrxreconximperial")

    # Init model and optimizer
    if config.resume_path is not None:
        print("Loading model from", config.resume_path)
        model, step = load(config, config.resume_path)
    else:
        model = get_model_from_name(config.model, config)
        step = 0

    optimizer = partial(torch.optim.AdamW, lr=config.lr, betas=config.adam_betas)
    model = EnsembleFactory(model, config.number_ensemble_models, "INDEPENDENT")
    optimizer = EnsembleOptimiserWrapper(model.ensemble_parameters(), optimizer)

    model.to(config.device)
    model.train()

    # print and log number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", num_params)
    logger.summary["num_params"] = num_params

    scaler = GradScaler()

    loss_function = get_loss_function_from_name(config.loss_function)

    # Load data
    dataloaders = build_dataloaders(
        data_dir=config.data_dir,
        acceleration_factor=config.acc_factor,
        val_split=config.val_split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pad_crop_slices=config.pad_crop_slices,
        pad_kspace=config.pad_kspace,
        single_slice=config.single_slice,
        number_coils=config.coils,
        debug=config.debug,
    )
    train_dl = dataloaders["train"]
    val_dl = dataloaders["val"]
    test_dl = dataloaders["test"] if "test" in dataloaders else None

    lr_scheduler = partial(
        torch.optim.lr_scheduler.CosineAnnealingLR,
        T_max=len(dataloaders["train"]) * config.epochs // config.scheduler_warmup_frequency,
        eta_min=1e-6,
    )
    lr_scheduler = EnsembleSchedulerWrapper(lr_scheduler, optimizer)

    if config.resume_path is not None:
        checkpoint = torch.load(config.resume_path, map_location=torch.device(config.device))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.last_epoch = checkpoint["step"]

    model = train(
        model=model,
        dataloader_train=train_dl,
        dataloader_valid=val_dl,
        loss_fn=loss_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        logger=logger,
        config=config,
    )

    if test_dl is not None:
        val_metrics = validation(val_dl, model, loss_function, config.device, task_name="val")
        logger.log(val_metrics)
        test_metrics = validation(test_dl, model, loss_function, config.device, task_name="test")
        logger.log(test_metrics)

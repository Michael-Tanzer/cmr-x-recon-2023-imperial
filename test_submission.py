from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
import os
from pathlib import Path
import torch
from dataloading import build_test_dataloader
from ensemble.factory import EnsembleFactory
from utils import get_model_from_name, seed_everything, set_custom_repr_tensors, tree
from torch import nn
from scipy.io import savemat
import numpy as np


def load(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    config = checkpoint["config"]
    model = get_model_from_name(config.model, config)
    step = checkpoint["step"]
    return model, step, config


@torch.no_grad()
def test(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    config: Namespace,
    debug: bool = False,
):
    device = config.device
    model.eval()
    inputs = []
    outputs = []

    for ii, (x, _) in enumerate(dataloader):
        x = x.to(device)

        x_kspaces = []
        x_kspace_masks = []

        if "t1" in config.tasks:
            x_kspaces.append(x.t1_kspace)
            x_kspace_masks.append(x.t1_kspace_mask)

        if "t2" in config.tasks:
            x_kspaces.append(x.t2_kspace)
            x_kspace_masks.append(x.t2_kspace_mask)

        y_pred = {}
        for name, x_kspace, x_kspace_mask in zip(config.tasks, x_kspaces, x_kspace_masks):
            output = model(x_kspace, x_kspace_mask, y_kspace=None, name=name)
            y_pred[f"{name}_output"] = output["output"]

        outputs.append({k: v.cpu() for k, v in y_pred.items()})
        inputs.append(x.to("cpu"))

    batched_outputs = []

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

    # flatten the inputs
    new_inputs = None
    for input in inputs:
        if new_inputs is None:
            new_inputs = deepcopy(input)
        else:
            for k, v in input.__dict__.items():
                if isinstance(v, torch.Tensor):
                    new_inputs.__dict__[k] = torch.cat([new_inputs.__dict__[k], v], dim=0)
                elif v is None:
                    new_inputs.__dict__[k] = None
                elif isinstance(v, bool) or isinstance(v, int) or isinstance(v, float):
                    new_inputs.__dict__[k] = v
                else:
                    new_inputs.__dict__[k].extend(v)
    inputs = new_inputs

    i = 0
    while (outputs[f"{name}_output"].shape[0] - i) > 0:
        # the index of the inputs.original_number_slices follows that every slice has a feature of original_number_slices.
        number_slices = inputs.original_number_slices[i].item()
        new_batched_outputs = {}
        for k, v in outputs.items():
            new_batched_outputs[k] = v[i : i + number_slices]
        batched_outputs.append(new_batched_outputs)
        i += number_slices

    outputs = batched_outputs

    # next step is saving the data as mat files
    if debug:
        save_dir = Path("test_outputs")
    else:
        save_dir = Path("/output")

    if config.coils == "single":
        save_dir = save_dir / "SingleCoil"
    else:
        save_dir = save_dir / "MultiCoil"

    save_dir = save_dir / "Mapping" / "TestSet" / f"AccFactor{config.acc_factor:02d}"

    save_dir.mkdir(exist_ok=True, parents=True)

    for i, output in enumerate(outputs):
        p_save_dir = save_dir / f"P{i + 1:03d}"
        p_save_dir.mkdir(exist_ok=True)

        task = config.tasks[0]
        data_to_save = output[f"{task}_output"].numpy()

        #if config.coils == "single": multi needs squeeze the 2nd dim as well. 
        data_to_save = data_to_save.squeeze(2)
        # final saving should be [nframe,nslice,(ncoil), height, width], so exchange [1,0]
        data_to_save = np.moveaxis(data_to_save, 0, 1)

        # save the data into a mat file with name "{task}map.mat" in the variable "img4ranking"
        savemat(p_save_dir / f"{task.upper()}map.mat", {"img4ranking": data_to_save})

    return outputs


def main(debug=False):
    set_custom_repr_tensors()
    seed_everything(999)

    # print the local structure
    for t in tree(Path(".")):
        print(t)

    gpu_available = torch.cuda.is_available()
    device = torch.device("cuda" if gpu_available else "cpu")

    for coils in ["single", "multi"]:
        for t1_t2 in ["t1", "t2"]:
            for acc_factor in [4, 8, 10]:
                if debug:
                    model_path = f"model_checkpoints/{coils}_{t1_t2}_{acc_factor}.pt"
                else:
                    model_path = f"/model_checkpoints/{coils}_{t1_t2}_{acc_factor}.pt"

                if not os.path.exists(model_path):
                    print("Model not found:", model_path, "Skipping")
                    continue

                print("Loading model from", model_path)
                model, step, config = load(model_path)

                if debug:
                    config.data_dir = "/home/mt3019/biomedia/vol/biodata/data/CMRxRecon 2023"
                else: 
                    config.data_dir = "/input"

                config.resume_path = model_path
                config.device = device

                model = EnsembleFactory(model, config.number_ensemble_models, "INDEPENDENT")

                model.to(config.device)
                model.eval()

                loss_function = lambda *args, **kwargs: 0

                # Load data
                test_dl = build_test_dataloader(
                    data_dir=config.data_dir,
                    acceleration_factor=config.acc_factor,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    pad_crop_slices=config.pad_crop_slices,
                    pad_kspace=config.pad_kspace,
                    single_slice=config.single_slice,
                    number_coils=config.coils,
                    debug=debug,
                )

                checkpoint = torch.load(config.resume_path, map_location=torch.device(config.device))
                model.load_state_dict(checkpoint["model_state_dict"])

                test_data = test(test_dl, model, config, debug=debug)

                del (
                    model,
                    test_dl,
                    checkpoint,
                    loss_function,
                    config,
                    model_path,
                    step,
                    test_data,
                )


if __name__ == "__main__":
    if os.environ.get("DEBUG"):
        main(debug=True)
    else:
        main(debug=False)

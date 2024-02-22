from copy import copy, deepcopy
from dataclasses import dataclass
from functools import partial
import logging
from pathlib import Path
import socket
from threading import Thread
import time
from typing import Callable, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler, Sampler
from torch import nn
from torchvision import transforms
import os
import h5py
import nibabel as nib
from nibabel.imageglobals import LoggingOutputSuppressor
from tqdm import tqdm
from dataobject import MappingDatapoint, custom_object_collate_fn
from utils import tree


class RandomMaskTransform(nn.Module):
    def __init__(self, acceleration_factor: int):
        super().__init__()
        self.acceleration_factor = acceleration_factor

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        number_lines = mask.shape[0]
        new_mask = torch.zeros_like(mask)
        starting_point = np.random.randint(0, self.acceleration_factor)
        new_mask[starting_point :: self.acceleration_factor] = 1
        new_mask[number_lines // 2 - 12 : number_lines // 2 + 12] = 1

        return new_mask.bool()


def get_train_indices():
    # exclude from black formatting
    # fmt: off
    return [ 
        26, 118, 90, 73, 50, 63, 97, 57, 35, 44, 84, 13, 23, 40,
        86, 66, 16,  9, 12, 19,  5, 49, 64, 93, 101, 65, 71, 41,
        61, 55, 28, 39, 56, 24, 75, 96, 22, 32, 87, 100, 114, 116,
        109, 82, 47, 111, 102,  3, 119, 25, 29, 110, 60, 17, 79, 14,
        113,  7, 81, 74, 1, 43, 85, 67, 89, 98, 112, 34, 30, 104,
        8, 33,  6, 45, 58, 10, 94, 48, 91, 38, 70, 72, 20, 11,
        52, 59, 46, 80,  4, 88, 15, 76, 51, 106, 77, 83, 95, 117,
        53, 99
    ]
    # fmt: on


def get_val_indices():
    return [0, 68, 115, 62, 54, 31, 42, 103, 21, 27]


def get_test_indices():
    return [78, 18, 107, 2, 69, 108, 105, 36, 37, 92]


def load_mapping_files(
    folder: str, roi_folder: str, pad_crop_slices: int = -1, pad_kspace: bool = False, number_coils: str = "single"
) -> Optional[MappingDatapoint]:
    mapping_data = {}
    for t1_t2 in ["T1", "T2"]:
        try:
            mapping_data |= {
                f"{t1_t2}_times": pd.read_csv(os.path.join(folder, f"{t1_t2}map.csv")),
                f"{t1_t2}_kspace": h5py.File(os.path.join(folder, f"{t1_t2}map.mat"), "r"),
            }
        except FileNotFoundError:
            return None

        try:
            mapping_data[f"{t1_t2}_kspace_mask"] = h5py.File(os.path.join(folder, f"{t1_t2}map_mask.mat"), "r")
        except FileNotFoundError:
            pass

        for k, v in mapping_data.items():
            if isinstance(v, pd.DataFrame):
                times_data = v.to_numpy()[:, 1:].astype(float)  # removes the first column with the index
                # remove the columns that are all nans
                times_data = times_data[:, ~np.all(np.isnan(times_data), axis=0)]
                mapping_data[k] = torch.tensor(times_data, dtype=torch.int32)
            elif isinstance(v, h5py.File):
                keys = list(v.keys())
                assert len(keys) == 1, f"Expected only one key in {k} file, got {len(keys)} instead"
                new_value = v[keys[0]][()]

                if new_value.dtype.names is not None and "real" in new_value.dtype.names:
                    # complex data, store it as complex tensor
                    real = new_value["real"]
                    imag = new_value["imag"]
                    new_value = torch.tensor(real + 1j * imag, dtype=torch.complex64)
                else:
                    new_value = torch.tensor(new_value, dtype=torch.float32)
                    # check if it only contains 0 and 1
                    if new_value.unique().tolist() == [0, 1]:
                        new_value = new_value.bool()

                mapping_data[k] = new_value

    mapping_data |= load_roi_files(roi_folder)

    mapping_data = {k.lower(): v for k, v in mapping_data.items()}

    n_slices = mapping_data["t1_times"].shape[1]
    if pad_crop_slices > 0:
        for k, v in mapping_data.items():
            if isinstance(v, torch.Tensor) and "times" in k:
                new_times = torch.zeros((v.shape[0], pad_crop_slices), dtype=v.dtype)
                new_times[:, : v.shape[1]] = v[:, :pad_crop_slices]
                mapping_data[k] = new_times

                if v.shape[1] > pad_crop_slices:
                    logging.warning(f"Times for {k} are longer than {pad_crop_slices} slices, cropping them")

            elif isinstance(v, torch.Tensor) and "kspace" in k and "mask" not in k:
                new_kspace = torch.zeros((v.shape[0], pad_crop_slices, *v.shape[2:]), dtype=v.dtype)
                new_kspace[:, : v.shape[1]] = v[:, :pad_crop_slices]
                mapping_data[k] = new_kspace

                if v.shape[1] > pad_crop_slices:
                    logging.warning(f"Kspace for {k} are longer than {pad_crop_slices} slices, cropping them")

            elif isinstance(v, torch.Tensor) and "map" in k:
                new_map = torch.zeros((v.shape[0], v.shape[1], pad_crop_slices), dtype=v.dtype)
                new_map[:, :, : v.shape[2]] = v[:, :, :pad_crop_slices]
                mapping_data[k] = new_map

                if v.shape[2] > pad_crop_slices:
                    logging.warning(f"Map for {k} are longer than {pad_crop_slices} slices, cropping them")

    dimensions_dont_match = (
        mapping_data["t1_kspace"].shape[2] != mapping_data["t2_kspace"].shape[2]
        or mapping_data["t1_kspace"].shape[3] != mapping_data["t2_kspace"].shape[3]
    )
    if dimensions_dont_match and pad_kspace:
        # pad the smaller one with zeros, centering the data
        t1_shape = mapping_data["t1_kspace"].shape
        t2_shape = mapping_data["t2_kspace"].shape

        height_diff_t1 = t2_shape[2] - t1_shape[2] if t1_shape[2] < t2_shape[2] else 0
        width_diff_t1 = t2_shape[3] - t1_shape[3] if t1_shape[3] < t2_shape[3] else 0
        height_diff_t2 = t1_shape[2] - t2_shape[2] if t2_shape[2] < t1_shape[2] else 0
        width_diff_t2 = t1_shape[3] - t2_shape[3] if t2_shape[3] < t1_shape[3] else 0

        padder_t1 = transforms.Pad(
            (
                width_diff_t1 // 2,
                height_diff_t1 // 2,
                width_diff_t1 - width_diff_t1 // 2,
                height_diff_t1 - height_diff_t1 // 2,
            )
        )
        mask_padder_t1 = transforms.Pad(
            (
                width_diff_t1 // 2,
                height_diff_t1 // 2,
                width_diff_t1 - width_diff_t1 // 2,
                height_diff_t1 - height_diff_t1 // 2,
            ),
            True,
        )
        padder_t2 = transforms.Pad(
            (
                width_diff_t2 // 2,
                height_diff_t2 // 2,
                width_diff_t2 - width_diff_t2 // 2,
                height_diff_t2 - height_diff_t2 // 2,
            )
        )
        mask_padder_t2 = transforms.Pad(
            (
                width_diff_t2 // 2,
                height_diff_t2 // 2,
                width_diff_t2 - width_diff_t2 // 2,
                height_diff_t2 - height_diff_t2 // 2,
            ),
            True,
        )

        mapping_data["t1_kspace"] = padder_t1(mapping_data["t1_kspace"])
        mapping_data["t2_kspace"] = padder_t2(mapping_data["t2_kspace"])

        try:
            mapping_data["t1_kspace_mask"] = mask_padder_t1(mapping_data["t1_kspace_mask"])
            mapping_data["t2_kspace_mask"] = mask_padder_t2(mapping_data["t2_kspace_mask"])
        except KeyError:
            pass

        try:
            mapping_data["t1_map"] = padder_t1(mapping_data["t1_map"])
            mapping_data["t1_map_roi"] = padder_t1(mapping_data["t1_map_roi"])
            mapping_data["t2_map"] = padder_t2(mapping_data["t2_map"])
            mapping_data["t2_map_roi"] = padder_t2(mapping_data["t2_map_roi"])
        except TypeError:
            pass

    mapping_data["number_slices"] = n_slices
    mapping_data["multi_coil"] = number_coils == "multi"
    mapping_data = MappingDatapoint(**{k: v for k, v in mapping_data.items()})
    return mapping_data


def load_roi_files(folder: str) -> Dict[str, torch.Tensor]:
    try:
        with LoggingOutputSuppressor():
            t1_map_roi = nib.load(os.path.join(folder, "T1map_label.nii.gz"))
            t2_map_roi = nib.load(os.path.join(folder, "T2map_label.nii.gz"))
            t1_map = nib.load(os.path.join(folder, "T1map_forlabel.nii.gz"))
            t2_map = nib.load(os.path.join(folder, "T2map_forlabel.nii.gz"))

    except FileNotFoundError:
        return {
            "t1_map_roi": None,
            "t2_map_roi": None,
            "t1_map": None,
            "t2_map": None,
        }

    return {
        "t1_map_roi": torch.tensor(t1_map_roi.get_fdata(), dtype=torch.int32),
        "t2_map_roi": torch.tensor(t2_map_roi.get_fdata(), dtype=torch.int32),
        "t1_map": torch.tensor(t1_map.get_fdata(), dtype=torch.float32),
        "t2_map": torch.tensor(t2_map.get_fdata(), dtype=torch.float32),
    }


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class MappingSingleCoilDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        training_stage: str,
        acceleration_factor: Literal[4, 8, 10],
        transforms: Callable = None,
        progress_bar: bool = True,
        pad_crop_slices: int = -1,
        pad_kspace: bool = False,
        single_slice: bool = False,
        number_coils: Literal["single", "multi"] = "single",
        debug: bool = False,
    ) -> None:
        if "mt3019" not in socket.gethostname():
           for line in tree(Path(root_dir)):
               print(line)
               
        # for test submission, no ChallengeData is included in the directory.
        if training_stage != "test" and "ChallengeData" not in root_dir:
           root_dir = os.path.join(root_dir, "ChallengeData")

        if "SingleCoil" not in root_dir and number_coils == "single":
            root_dir = os.path.join(root_dir, "SingleCoil")
        elif "MultiCoil" not in root_dir and number_coils == "multi":
            root_dir = os.path.join(root_dir, "MultiCoil")

        if "Mapping" not in root_dir:
            root_dir = os.path.join(root_dir, "Mapping")

        self.root_dir = root_dir
        self.training_stage = training_stage
        self.acceleration_factor = acceleration_factor
        self.transforms = transforms
        self.single_slice = single_slice
        self.number_coils = number_coils

        self.MAX_QUEUE_SIZE = 25
        self.data = []
        self.functions = []
        self.next_indices = []
        self.number_slices = []
        self.threads = []

        if self.training_stage == "train":
            self.root_dir = os.path.join(self.root_dir, f"TrainingSet")
        elif self.training_stage == "val":
            self.root_dir = os.path.join(self.root_dir, f"ValidationSet")
        elif self.training_stage == "test":
            self.root_dir = os.path.join(self.root_dir, f"TestSet")
        else:
            raise ValueError(f"Unknown training stage {self.training_stage}")

        data_dir = os.path.join(self.root_dir, f"AccFactor{self.acceleration_factor:02d}")
        target_dir = os.path.join(self.root_dir, "FullSample")
        segment_dir = os.path.join(self.root_dir, "SegmentROI")

        # iterate over first level sub folders in target_dir
        for root, dirs, files in os.walk(data_dir):
            if not dirs:
                continue

            for ii, dir in enumerate(tqdm(list(sorted(dirs)), "Loading data", disable=not progress_bar)):
                mapping_input, mapping_target = self._load_data(
                    dir,
                    data_dir,
                    target_dir,
                    segment_dir,
                    pad_crop_slices,
                    pad_kspace,
                )

                if self.training_stage == "train" and any(x is None for x in [mapping_input, mapping_target]):
                    continue
                elif not self.training_stage == "train" and mapping_input is None:
                    continue

                self.number_slices.append(mapping_input.number_slices)

                self.functions.append((dir, data_dir, target_dir, segment_dir, pad_crop_slices, pad_kspace))

                if self.MAX_QUEUE_SIZE < 0:
                    self.data.append((mapping_input, mapping_target))
                else:
                    self.data.append(None)

                if ii == 5 and debug:
                    break

        self.threads = [None for _ in range(len(self))]

    def _load_data(
        self,
        dir: str,
        data_dir: str,
        target_dir: str,
        segment_dir: str,
        pad_crop_slices: int,
        pad_kspace: bool,
    ):
        try:
            # raise FileNotFoundError
            mapping_input = torch.load(os.path.join(data_dir, dir, "data.pth"))
        except (FileNotFoundError, RuntimeError):
            mapping_input = load_mapping_files(
                os.path.join(data_dir, dir),
                os.path.join(segment_dir, dir),
                pad_crop_slices,
                pad_kspace,
                number_coils=self.number_coils,
            )
            torch.save(mapping_input, os.path.join(data_dir, dir, "data.pth"))

        try:
            # raise FileNotFoundError
            mapping_target = torch.load(os.path.join(target_dir, dir, "data.pth"))
        except (FileNotFoundError, RuntimeError):
            mapping_target = load_mapping_files(
                os.path.join(target_dir, dir),
                os.path.join(segment_dir, dir),
                pad_crop_slices,
                pad_kspace,
                number_coils=self.number_coils,
            )
            if mapping_target is not None:
                torch.save(mapping_target, os.path.join(target_dir, dir, "data.pth"))

        return mapping_input, mapping_target

    def __len__(self):
        if self.single_slice:
            return sum(self.number_slices)
        else:
            return len(self.data)

    def __getitem__(self, idx) -> Tuple[MappingDatapoint, MappingDatapoint]:
        dataset_index, slice_index = self.get_true_indices(idx)

        # I need to add to the queue if the queue length is < 10, not if it's empty
        number_non_none_threads = len([t for t in self.threads if t is not None])
        if number_non_none_threads < self.MAX_QUEUE_SIZE:
            for i in range(self.MAX_QUEUE_SIZE - number_non_none_threads):
                if not self.next_indices:
                    break

                next_index = self.next_indices.pop(0)
                next_dataset_index, next_slice_index = self.get_true_indices(next_index)

                function = self.functions[next_dataset_index]
                t = ThreadWithReturnValue(target=self._load_data, args=function)
                t.start()
                self.threads[next_index] = t

        if self.threads[idx] is None and self.MAX_QUEUE_SIZE >= 0:
            function = self.functions[dataset_index]
            t = ThreadWithReturnValue(target=self._load_data, args=function)
            t.start()
            self.threads[idx] = t
            if idx in self.next_indices:
                self.next_indices.remove(idx)
            print(
                "current idx was not ready, starting thread",
                idx,
                dataset_index,
                slice_index,
                number_non_none_threads,
                len(self.next_indices),
            )

        if self.MAX_QUEUE_SIZE >= 0:
            mapping_input, mapping_target = self.threads[idx].join()
            self.threads[idx] = None
        else:
            mapping_input, mapping_target = self.data[dataset_index]

        if self.single_slice:
            mapping_input = mapping_input.select_slice(slice_index)
            mapping_target = mapping_target.select_slice(slice_index) if mapping_target is not None else None

        if self.transforms is not None and mapping_target is not None:
            mapping_input.t1_kspace_mask = self.transforms(mapping_input.t1_kspace_mask)
            mapping_input.t2_kspace_mask = self.transforms(mapping_input.t2_kspace_mask)

            mapping_input.t1_kspace = mapping_target.t1_kspace.clone()
            mapping_input.t2_kspace = mapping_target.t2_kspace.clone()

            mapping_input.t1_kspace[..., ~mapping_input.t1_kspace_mask] = 0
            mapping_input.t2_kspace[..., ~mapping_input.t2_kspace_mask] = 0

        return mapping_input, mapping_target

    def get_true_indices(self, idx):
        dataset_index, slice_index = 0, 0
        if self.single_slice:
            for num_slices in self.number_slices:
                if idx < num_slices:
                    slice_index = idx
                    break
                else:
                    idx -= num_slices
                    dataset_index += 1
            else:
                raise IndexError(f"Index {idx} out of range")
        else:
            dataset_index = idx
            slice_index = None

        return dataset_index, slice_index

    def set_next_indices(self, indices):
        # self.next_indices = [self.get_true_indices(idx)[0] for idx in indices]
        self.next_indices = indices

    def set_threads(self):
        self.threads = [None for _ in range(len(self))]


class BatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer value, " "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got " "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.reset_batches()

    def reset_batches(self):
        self.batches = []
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    self.batches.append(batch)
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    self.batches.append(batch)
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                self.batches.append(batch[:idx_in_batch])

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.batches:
            yield batch

        self.reset_batches()

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


def build_dataloaders(
    data_dir: str,
    acceleration_factor: Literal[4, 8, 10],
    val_split: float,
    batch_size: int,
    num_workers: int,
    pad_crop_slices: int = -1,
    pad_kspace: bool = False,
    single_slice: bool = False,
    number_coils: Literal["single", "multi"] = "single",
    debug: bool = False,
):
    if number_coils == "single":
        transforms_train = transforms.Compose(
            [
                RandomMaskTransform(acceleration_factor),
            ]
        )
    else:
        transforms_train = transforms.Compose([])

    transforms_test = transforms.Compose([])

    train_val_dataset = MappingSingleCoilDataset(
        data_dir,
        "train",
        acceleration_factor,
        transforms=transforms_train,
        progress_bar=True,
        pad_crop_slices=pad_crop_slices,
        single_slice=single_slice,
        pad_kspace=pad_kspace,
        number_coils=number_coils,
        debug=debug,
    )

    train_dataset, val_dataset, test_dataset = (
        deepcopy(train_val_dataset),
        deepcopy(train_val_dataset),
        deepcopy(train_val_dataset),
    )

    indices_train = get_train_indices()
    indices_val = get_val_indices()
    indices_test = get_test_indices()

    indices_train = [i for i in indices_train if i < len(train_val_dataset.data)]
    indices_val = [i for i in indices_val if i < len(train_val_dataset.data)]
    indices_test = [i for i in indices_test if i < len(train_val_dataset.data)]

    train_dataset.data = [train_val_dataset.data[i] for i in indices_train]
    train_dataset.functions = [train_val_dataset.functions[i] for i in indices_train]
    train_dataset.number_slices = [train_val_dataset.number_slices[i] for i in indices_train]
    val_dataset.data = [train_val_dataset.data[i] for i in indices_val]
    val_dataset.functions = [train_val_dataset.functions[i] for i in indices_val]
    val_dataset.number_slices = [train_val_dataset.number_slices[i] for i in indices_val]
    test_dataset.data = [train_val_dataset.data[i] for i in indices_test]
    test_dataset.functions = [train_val_dataset.functions[i] for i in indices_test]
    test_dataset.number_slices = [train_val_dataset.number_slices[i] for i in indices_test]

    val_dataset.transforms = transforms_test
    test_dataset.transforms = transforms_test

    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=batch_size, drop_last=False)
    val_sampler = BatchSampler(SequentialSampler(val_dataset), batch_size=batch_size, drop_last=False)
    test_sampler = BatchSampler(SequentialSampler(test_dataset), batch_size=batch_size, drop_last=False)

    # flat batches from the sampler
    train_dataset.set_next_indices([i for batch in train_sampler.batches for i in batch])
    val_dataset.set_next_indices([i for batch in val_sampler.batches for i in batch])
    test_dataset.set_next_indices([i for batch in test_sampler.batches for i in batch])

    train_dataset.set_threads()
    val_dataset.set_threads()
    test_dataset.set_threads()

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_object_collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_object_collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_object_collate_fn,
    )

    return {"train": train_dataloader, "test": test_dataloader, "val": val_dataloader}


def build_test_dataloader(
    data_dir: str,
    acceleration_factor: Literal[4, 8, 10],
    batch_size: int,
    num_workers: int,
    pad_crop_slices: int = -1,
    pad_kspace: bool = False,
    single_slice: bool = False,
    number_coils: Literal["single", "multi"] = "single",
    debug: bool = False,
):
    transforms_test = transforms.Compose([])
    test_dataset = MappingSingleCoilDataset(
        data_dir,
        "val" if debug else "test",
        acceleration_factor,
        transforms=transforms_test,
        progress_bar=True,
        pad_crop_slices=pad_crop_slices,
        single_slice=single_slice,
        pad_kspace=pad_kspace,
        number_coils=number_coils,
        debug=False,
    )

    test_sampler = BatchSampler(SequentialSampler(test_dataset), batch_size=batch_size, drop_last=False)
    test_dataset.set_next_indices([i for batch in test_sampler.batches for i in batch])
    test_dataset.set_threads()

    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_object_collate_fn,
    )

    return test_dataloader


if __name__ == "__main__":
    folder = "/home/mt3019/biomedia/vol/biodata/data/CMRxRecon 2023"

    dataloaders = build_dataloaders(
        folder,
        4,
        0.2,
        4,
        0,
        single_slice=True,
        number_coils="single",
        debug=False,
    )

    print(dataloaders["train"].dataset.next_indices)
    print(dataloaders["train"].batch_sampler.batches)

    for x, y in dataloaders["train"]:
        print(x.number_slices, y.number_slices)

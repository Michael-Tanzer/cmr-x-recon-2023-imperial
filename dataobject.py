from dataclasses import dataclass
from typing import List, Optional
import torch
from torch.utils.data import default_collate


@dataclass
class MappingDatapoint:
    t1_times: Optional[torch.Tensor]
    t2_times: Optional[torch.Tensor]
    t1_kspace: Optional[torch.Tensor]
    t2_kspace: Optional[torch.Tensor]

    multi_coil: bool

    t1_map: Optional[torch.Tensor]
    t2_map: Optional[torch.Tensor]
    t1_map_roi: Optional[torch.Tensor]
    t2_map_roi: Optional[torch.Tensor]

    t1_kspace_mask: Optional[torch.BoolTensor] = None
    t2_kspace_mask: Optional[torch.BoolTensor] = None

    num_dims: int = 4
    number_slices: Optional[int] = None
    original_number_slices: Optional[int] = None

    def __post_init__(self):
        if self.multi_coil:
            self.num_dims = 5
        else:
            self.num_dims = 4

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        elif isinstance(key, int):
            return self.select_slice(key)
        elif isinstance(key, slice):
            start = key.start
            stop = key.stop
            step = key.step

            if start is None:
                start = 0

            if stop is None and self.t1_kspace.ndim == self.num_dims:
                stop = self.number_slices
            elif stop is None and self.t1_kspace.ndim == self.num_dims + 1:  # batched
                stop = self.t1_kspace.shape[0]
            elif stop is None:
                raise ValueError("the number of dimensions of the kspace is not supported")

            if step is None:
                step = 1

            new_dict = {}

            for k, v in self.__dict__.items():
                if isinstance(v, torch.Tensor) and ("times" in k or ("kspace" in k and "mask" not in k)):
                    if self.t1_kspace.ndim == self.num_dims:
                        v = v[:, start:stop:step]
                    elif self.t1_kspace.ndim == self.num_dims + 1:
                        v = v[start:stop:step]
                    new_dict[k] = v

                elif isinstance(v, torch.Tensor) and "kspace" in k and "mask" in k:
                    if self.t1_kspace.ndim == 5:
                        v = v[start:stop:step]
                    new_dict[k] = v

                elif isinstance(v, torch.Tensor) and "map" in k:
                    if self.t1_kspace.ndim == self.num_dims:
                        v = v[:, :, start:stop:step]
                    elif self.t1_kspace.ndim == self.num_dims + 1:
                        v = v[start:stop:step]
                    new_dict[k] = v

                elif k == "number_slices" and self.t1_kspace.ndim == self.num_dims:
                    new_dict[k] = (stop - start) // step
                    new_dict["original_number_slices"] = v

                elif k == "number_slices" and self.t1_kspace.ndim == self.num_dims + 1:
                    new_dict[k] = v[start:stop:step]

                elif k == "original_number_slices" and isinstance(v, int):
                    new_dict[k] = v

                elif k == "original_number_slices" and isinstance(v, torch.Tensor):
                    new_dict[k] = v[start:stop:step]

                elif v is None or isinstance(v, bool) or isinstance(v, int) or isinstance(v, float):
                    new_dict[k] = v

                else:
                    raise ValueError("key is not supported")

            return MappingDatapoint(**new_dict)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __iter__(self):
        for k, v in self.__dict__.items():
            yield k, v

    def __len__(self):
        if self.t1_kspace.ndim == self.num_dims + 1:
            return self.t1_kspace.shape[0]
        elif self.t1_kspace.ndim == self.num_dims:
            return self.number_slices

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))

        return self

    def select_slice(self, slice_index: int):
        new_dict = {}

        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor) and ("times" in k or ("kspace" in k and "mask" not in k)):
                if "times" in k and v.shape[1] == 1:
                    v = v[:, 0:1]
                else:
                    v = v[:, slice_index : slice_index + 1]
                new_dict[k] = v

            elif isinstance(v, torch.Tensor) and "map" in k:
                v = v[:, :, slice_index : slice_index + 1]
                new_dict[k] = v

            elif k == "number_slices":
                new_dict[k] = 1
                new_dict["original_number_slices"] = v

            elif k != "original_number_slices":
                new_dict[k] = v

            elif k == "multi_coil" or k == "num_dims":
                new_dict[k] = v

        return MappingDatapoint(**new_dict)


def custom_object_collate_fn(data_points: List[MappingDatapoint]):
    inputs = [d[0] for d in data_points]
    targets = [d[1] for d in data_points]

    input_dict = {}
    for k in inputs[0].__dict__:
        new_values = []
        for point in inputs:
            new_values.append(getattr(point, k))

        if all([v is None for v in new_values]):
            input_dict[k] = None
        elif k == "multi_coil" or k == "num_dims":
            input_dict[k] = new_values[0]
        else:
            input_dict[k] = default_collate(new_values)

    inputs = MappingDatapoint(**input_dict)

    if targets[0] is None:
        targets = None
    else:
        target_dict = {}
        for k in targets[0].__dict__:
            new_values = []
            for point in targets:
                new_values.append(getattr(point, k))

            if all([v is None for v in new_values]):
                target_dict[k] = None
            elif k == "multi_coil" or k == "num_dims":
                target_dict[k] = new_values[0]
            else:
                target_dict[k] = default_collate(new_values)

        targets = MappingDatapoint(**target_dict)

    return inputs, targets

from abc import abstractmethod
from enum import Enum

from torch import nn


class EnsembleType(Enum):
    INDEPENDENT = "INDEPENDENT"


class BaseEnsemble:
    def __init__(self, model_generator_function, models_n, isgan=False):
        assert models_n > 0, "Can't train an ensemble with 0 or fewer models"

        self.models_n = models_n
        self.isgan = isgan

        self.models = nn.ModuleList(modules=[model_generator_function() for _ in range(models_n)])
        self.current_model_idx = 0

        self.using_cuda = False
        self.cuda_device = None
        self.to_what = None
        self.training = True

    def parameters(self, recurse: bool = True):
        return self.models.parameters(recurse=recurse)

    def ensemble_parameters(self, recurse: bool = True):
        parameters = {}

        for i in range(self.models_n):
            pars = self.models[i].parameters(recurse=recurse)
            if isinstance(pars, dict):
                pars = {k: list(v) for k, v in pars.items()}
            else:
                pars = list(pars)
            parameters[f"models.{i}"] = pars

        return parameters

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.models.named_parameters()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.models.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.models.load_state_dict(state_dict, strict=strict)

    def cuda(self, device=None):
        self.using_cuda = True
        self.cuda_device = device
        self.models[self.current_model_idx].cuda(self.cuda_device)

        return self

    def to(self, *args, **kwargs):
        self.to_what = (args, kwargs)
        self.models[self.current_model_idx].to(*self.to_what[0], **self.to_what[1])

        return self

    def train(self, mode: bool = True):
        self.training = mode
        self.models[self.current_model_idx].train(mode)

        return self

    def eval(self):
        self.training = False
        self.models[self.current_model_idx].eval()

        return self

    def start_next_model(self):
        self.current_model_idx += 1

        if self.using_cuda:
            self.models[self.current_model_idx - 1].cpu()
            self.models[self.current_model_idx].cuda(self.cuda_device)

        if self.to_what is not None:
            self.models[self.current_model_idx].to(*self.to_what[0], **self.to_what[1])

        if self.training:
            self.models[self.current_model_idx].train()
        else:
            self.models[self.current_model_idx].eval()

    def __getattr__(self, item):
        return getattr(self.models[self.current_model_idx], item)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class EnsembleOptimiserWrapper:
    def __init__(self, params, optimiser_generator_function):
        super().__init__()
        self.optimisers = []
        self.current_optimiser_idx = 0

        for k, parameters in params.items():
            self.optimisers.append(optimiser_generator_function(parameters))

    def start_next_model(self):
        self.current_optimiser_idx += 1

    def state_dict(self, **kwargs):
        states = {}
        for i in range(len(self.optimisers)):
            states[i] = self.optimisers[i].state_dict(**kwargs)
        return states

    def load_state_dict(self, state_dict, **kwargs):
        for k, sd in state_dict.items():
            self.optimisers[k].load_state_dict(sd, **kwargs)

    def __getattr__(self, item):
        return getattr(self.optimisers[self.current_optimiser_idx], item)


class EnsembleSchedulerWrapper:
    def __init__(self, scheduler_generator_function, optimizer):
        super().__init__()
        self.schedulers = []
        self.current_scheduler_idx = 0

        for i, optimizer in enumerate(optimizer.optimisers):
            self.schedulers.append(scheduler_generator_function(optimizer))

    def start_next_model(self):
        self.current_scheduler_idx += 1

    def state_dict(self, **kwargs):
        return {i: self.schedulers[i].state_dict(**kwargs) for i in range(len(self.schedulers))}

    def load_state_dict(self, state_dict, **kwargs):
        for k, sd in state_dict.items():
            self.schedulers[k].load_state_dict(sd, **kwargs)

    def __getattr__(self, item):
        return getattr(self.schedulers[self.current_scheduler_idx], item)

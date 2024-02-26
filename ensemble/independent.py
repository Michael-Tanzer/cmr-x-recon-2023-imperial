import torch

from ensemble.base import BaseEnsemble


class IndependentEnsemble(BaseEnsemble):
    def get_model(self, index):
        if self.isgan:
            return self.models[index].generator
        else:
            return self.models[index]

    def __call__(self, *args, **kwargs):
        if self.training:
            model_output = self.get_model(self.current_model_idx)(*args, **kwargs)

            return model_output
        else:
            original_current_model_device = next(self.get_model(self.current_model_idx).parameters()).device

            # only move to cpu if there are multiple models
            if self.current_model_idx > 0:
                self.get_model(self.current_model_idx).cpu()

            model_outputs = []
            for i in range(self.current_model_idx + 1):
                original_device = next(self.get_model(i).parameters()).device
                self.get_model(i).to(original_current_model_device)
                model_outputs.append(self.get_model(i)(*args, **kwargs))
                self.get_model(i).to(original_device)

            self.get_model(self.current_model_idx).to(original_current_model_device)

            var, mean = torch.var_mean(torch.stack([d["output"] for d in model_outputs]), dim=0)

            return_dictionary = {
                "output": mean,
                "variance": var.mean(1, keepdim=True),
            }

            if "loss" in model_outputs[0]:
                losses = sum([d["loss"] for d in model_outputs]) / len(model_outputs)
                return_dictionary["loss"] = losses

            return return_dictionary

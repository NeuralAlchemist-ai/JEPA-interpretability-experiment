from collections.abc import Callable
import torch
import torch.nn as nn


def capture_activations(module: nn.Module) -> tuple[nn.utils.hooks.RemovableHandle, list[torch.Tensor]]:
    captured = []

    def hook(mod, inputs, output):
        captured.append(output.detach().clone())

    handle = module.register_forward_hook(hook)
    return handle, captured


def register_intervention_hook(
    module: nn.Module, intervention_fn: Callable[[torch.Tensor], torch.Tensor]
) -> nn.utils.hooks.RemovableHandle:
    def modifying_hook(mod, inputs, output):
        return intervention_fn(output)

    return module.register_forward_hook(modifying_hook)

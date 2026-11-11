import torch.nn as nn


def _is_trainable(name: str, prefixes: list[str]) -> bool:
    return any(name.startswith(p) for p in prefixes)


def freeze_all_but(model: nn.Module, trainable_prefixes: list[str]) -> list[str]:
    """Set requires_grad only for params matching a prefix; return their names."""
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad = _is_trainable(name, trainable_prefixes)
        if param.requires_grad:
            trainable.append(name)
    if not trainable:
        raise ValueError(f"No parameter matches prefixes {trainable_prefixes}")
    return trainable


def set_frozen_eval(model: nn.Module, trainable_prefixes: list[str]) -> None:
    """Put every submodule without a trainable param in eval() (call after train())."""
    for mod_name, module in model.named_modules():
        has_trainable = any(
            _is_trainable(f"{mod_name}.{n}" if mod_name else n, trainable_prefixes)
            for n, _ in module.named_parameters()
        )
        if not has_trainable:
            module.eval()

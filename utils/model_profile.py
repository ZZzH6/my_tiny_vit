from __future__ import annotations

import copy
from typing import Any

import torch

try:
    from thop import profile as thop_profile
except Exception:  # pragma: no cover - optional dependency
    thop_profile = None


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _format_note(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def profile_model(
    model: torch.nn.Module,
    input_size: tuple[int, int, int] = (3, 224, 224),
) -> dict[str, Any]:
    params = count_parameters(model)
    result: dict[str, Any] = {
        "params": params,
        "params_m": params / 1e6,
        "flops": None,
        "flops_g": None,
        "flops_status": "unavailable",
        "flops_note": "",
    }

    if thop_profile is None:
        result["flops_note"] = "thop is not installed"
        return result

    model_copy = None
    was_training = model.training
    try:
        model_copy = copy.deepcopy(model).cpu()
        model_copy.eval()
        dummy = torch.zeros((1, *input_size), device="cpu")
        with torch.no_grad():
            macs, _ = thop_profile(model_copy, inputs=(dummy,), verbose=False)
        flops = float(macs) * 2.0
        result["flops"] = int(round(flops))
        result["flops_g"] = flops / 1e9
        result["flops_status"] = "estimated"
        result["flops_note"] = "estimated with thop; FLOPs=2xMACs"
        return result
    except Exception as exc:  # pragma: no cover - defensive fallback
        result["flops_note"] = _format_note(exc)
        return result
    finally:
        if model_copy is not None:
            del model_copy
        if was_training:
            model.train()
        else:
            model.eval()

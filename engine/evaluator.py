from __future__ import annotations

import torch


def evaluate(model, loader, device):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            max_k = min(5, outputs.size(1))
            topk = outputs.topk(max_k, dim=1).indices
            top1_correct += (topk[:, 0] == targets).sum().item()
            top5_correct += (topk == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)

    if total == 0:
        return {"top1": 0.0, "top5": 0.0, "num_samples": 0}

    return {
        "top1": 100.0 * top1_correct / total,
        "top5": 100.0 * top5_correct / total,
        "num_samples": total,
    }

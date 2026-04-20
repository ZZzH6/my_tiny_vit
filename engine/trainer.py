from __future__ import annotations

import torch

from .distillation import compute_hard_distillation_loss, compute_soft_distillation_loss


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    mixup_fn=None,
    scaler=None,
    max_grad_norm=None,
    model_ema=None,
    teacher_model=None,
    distillation_alpha=0.0,
    distillation_temperature=1.0,
    distillation_method="logit",
    distillation_type="soft",
):
    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    distillation_alpha = float(distillation_alpha)
    if distillation_alpha < 0.0 or distillation_alpha > 1.0:
        raise ValueError(f"distillation_alpha must be in [0, 1], got {distillation_alpha}")
    distillation_method = str(distillation_method)
    distillation_type = str(distillation_type)

    total_loss = 0.0
    total_samples = 0
    use_amp = scaler is not None and scaler.is_enabled()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            if teacher_model is not None and distillation_alpha > 0.0:
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)

                if distillation_method == "deit":
                    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
                        raise ValueError(
                            "DeiT-style distillation expects model(images) to return "
                            "a tuple of (cls_logits, dist_logits) during training."
                        )
                    cls_outputs, dist_outputs = outputs
                    base_loss = criterion(cls_outputs, targets)
                    if distillation_type == "hard":
                        distillation_loss = compute_hard_distillation_loss(
                            student_logits=dist_outputs,
                            teacher_logits=teacher_outputs,
                        )
                    elif distillation_type == "soft":
                        distillation_loss = compute_soft_distillation_loss(
                            student_logits=dist_outputs,
                            teacher_logits=teacher_outputs,
                            temperature=distillation_temperature,
                        )
                    else:
                        raise ValueError(f"Unsupported distillation_type: {distillation_type}")
                else:
                    if isinstance(outputs, (tuple, list)):
                        raise ValueError(
                            "Logit distillation expects model(images) to return a single logits tensor."
                        )
                    base_loss = criterion(outputs, targets)
                    distillation_loss = compute_soft_distillation_loss(
                        student_logits=outputs,
                        teacher_logits=teacher_outputs,
                        temperature=distillation_temperature,
                    )
                loss = ((1.0 - distillation_alpha) * base_loss) + (distillation_alpha * distillation_loss)
            else:
                if isinstance(outputs, (tuple, list)):
                    if len(outputs) != 2:
                        raise ValueError(f"Unexpected training output tuple length: {len(outputs)}")
                    outputs = outputs[0]
                base_loss = criterion(outputs, targets)
                loss = base_loss

        if use_amp:
            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if model_ema is not None:
                model_ema.update(model)
        else:
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if model_ema is not None:
                model_ema.update(model)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples if total_samples > 0 else 0.0

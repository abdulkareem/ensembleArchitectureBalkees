import random
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_binary_output(logits: torch.Tensor, size=(256, 256)) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if logits.shape[1] > 1:
        logits = logits[:, :1, ...]
    if logits.shape[2:] != size:
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return logits


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = ensure_binary_output(logits, size=targets.shape[2:])
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + 1e-6) / (union + 1e-6)
    dice_loss = 1.0 - dice.mean()
    return 0.5 * bce + 0.5 * dice_loss


def compute_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    logits = ensure_binary_output(logits, size=targets.shape[2:])
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    t = (targets > 0.5).float()

    tp = ((preds == 1) & (t == 1)).sum().item()
    fp = ((preds == 1) & (t == 0)).sum().item()
    fn = ((preds == 0) & (t == 1)).sum().item()
    tn = ((preds == 0) & (t == 0)).sum().item()

    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    acc = (tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)
    return {"Dice": dice, "IoU": iou, "Precision": precision, "Recall": recall, "Accuracy": acc}


def robust_load_weights(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    model_state = model.state_dict()
    loaded_layers: List[str] = []
    skipped_layers: List[str] = []

    for k, v in ckpt.items():
        nk = k.replace("module.", "")
        if nk in model_state and model_state[nk].shape == v.shape:
            model_state[nk] = v
            loaded_layers.append(nk)
        else:
            skipped_layers.append(k)

    model.load_state_dict(model_state)
    print(f"[Loader] Loaded layers: {len(loaded_layers)}")
    print(f"[Loader] Skipped layers: {len(skipped_layers)}")
    if loaded_layers:
        print("[Loader] Sample loaded:", loaded_layers[:10])
    if skipped_layers:
        print("[Loader] Sample skipped:", skipped_layers[:10])


def parameter_stats(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_fps(model: nn.Module, device: torch.device, input_size=(1, 3, 256, 256), warmup=20, runs=100):
    model.eval()
    x = torch.randn(*input_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.time() - t0
    return runs / max(sec, 1e-6)

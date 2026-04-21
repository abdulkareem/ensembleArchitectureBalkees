from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from utils import compute_metrics_from_logits, ensure_binary_output


def evaluate_model(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    bag = {"Dice": [], "IoU": [], "Precision": [], "Recall": [], "Accuracy": []}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            metrics = compute_metrics_from_logits(model(x), y)
            for k, v in metrics.items():
                bag[k].append(v)
    return {k: float(np.mean(v)) for k, v in bag.items()}


def plot_training_curves(history: Dict[str, list], title: str):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        ax[0].plot(history["val_loss"], label="val_loss")
    ax[0].set_title(f"{title} Loss")
    ax[0].legend()

    ax[1].plot(history["val_dice"], label="val_dice")
    ax[1].set_title(f"{title} Dice")
    ax[1].legend()
    plt.show()


def visualize_predictions(models: Dict[str, torch.nn.Module], loader, device: torch.device, n_samples: int = 3):
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        preds = {name: torch.sigmoid(ensure_binary_output(model(x))).cpu() for name, model in models.items()}

    x_np = x.cpu().permute(0, 2, 3, 1).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x_np = np.clip(x_np * std + mean, 0, 1)

    for i in range(min(n_samples, x.shape[0])):
        cols = 2 + len(models)
        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
        axes[0].imshow(x_np[i])
        axes[0].set_title("Input")
        axes[1].imshow(y[i, 0].cpu().numpy(), cmap="gray")
        axes[1].set_title("Ground Truth")

        for j, (name, pred) in enumerate(preds.items(), start=2):
            axes[j].imshow((pred[i, 0].numpy() > 0.5).astype(np.float32), cmap="gray")
            axes[j].set_title(name)

        for a in axes:
            a.axis("off")
        plt.tight_layout()
        plt.show()


def build_comparison_table(metrics_by_model: Dict[str, Dict[str, float]], params_by_model: Dict[str, int], fps_by_model: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for name, metrics in metrics_by_model.items():
        rows.append({
            "Model": name,
            "Dice": metrics["Dice"],
            "IoU": metrics["IoU"],
            "Params": params_by_model.get(name, 0),
            "FPS": fps_by_model.get(name, 0.0),
        })
    return pd.DataFrame(rows).sort_values("Dice", ascending=False)


def plot_dice_bars(metrics_by_model: Dict[str, Dict[str, float]]):
    names = list(metrics_by_model.keys())
    dices = [metrics_by_model[n]["Dice"] for n in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, dices)
    plt.ylabel("Dice")
    plt.title("Model Dice Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.show()

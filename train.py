import os
from typing import Dict

import numpy as np
import torch

from utils import compute_metrics_from_logits, dice_bce_loss


def train_model(model, train_loader, val_loader, device: torch.device, epochs: int = 15, lr: float = 1e-4, save_name: str = "best_model.pth"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = dice_bce_loss(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_dices = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_losses.append(dice_bce_loss(logits, y).item())
                val_dices.append(compute_metrics_from_logits(logits, y)["Dice"])

        tr_loss = float(np.mean(train_losses))
        va_loss = float(np.mean(val_losses))
        va_dice = float(np.mean(val_dices))

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_dice"].append(va_dice)

        print(f"[{save_name}] Epoch {ep}/{epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_dice={va_dice:.4f}")

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(model.state_dict(), save_name)

    if os.path.exists(save_name):
        model.load_state_dict(torch.load(save_name, map_location=device))
    return history, save_name

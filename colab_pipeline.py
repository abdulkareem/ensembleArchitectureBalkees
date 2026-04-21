"""Colab-ready end-to-end training/evaluation pipeline for Kvasir-SEG."""

import argparse
import os

import pandas as pd
import torch

from dataset import DataConfig, make_dataloaders
from ensemble import WeightedEnsemble, train_ensemble_head
from evaluate import build_comparison_table, evaluate_model, plot_dice_bars, plot_training_curves, visualize_predictions
from models import ResUNetPPWrapper, TransFuse, WDFFNet
from train import train_model
from utils import measure_fps, parameter_stats, robust_load_weights, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Path to Kvasir-SEG root")
    p.add_argument("--resunet_source", type=str, default="/content/ResUNetPlusPlus/resunet++_pytorch.py")
    p.add_argument("--resunet_ckpt", type=str, default="")
    p.add_argument("--transfuse_ckpt", type=str, default="")
    p.add_argument("--wdff_ckpt", type=str, default="")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--ensemble_epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    data_cfg = DataConfig(data_dir=args.data_dir, image_size=256, batch_size=args.batch_size)
    train_loader, val_loader, test_loader = make_dataloaders(data_cfg)

    resunet = ResUNetPPWrapper(repo_file=args.resunet_source, out_size=256)
    transfuse = TransFuse(out_size=256, pretrained=True)
    wdff = WDFFNet(out_size=256, pretrained=True)

    if args.resunet_ckpt and os.path.exists(args.resunet_ckpt):
        robust_load_weights(resunet, args.resunet_ckpt, device)
    if args.transfuse_ckpt and os.path.exists(args.transfuse_ckpt):
        robust_load_weights(transfuse, args.transfuse_ckpt, device)
    if args.wdff_ckpt and os.path.exists(args.wdff_ckpt):
        robust_load_weights(wdff, args.wdff_ckpt, device)

    h_res, _ = train_model(resunet, train_loader, val_loader, device, epochs=args.epochs, save_name="best_resunetpp.pth")
    h_trf, _ = train_model(transfuse, train_loader, val_loader, device, epochs=args.epochs, save_name="best_transfuse.pth")
    h_wdf, _ = train_model(wdff, train_loader, val_loader, device, epochs=args.epochs, save_name="best_wdffnet.pth")

    ensemble = WeightedEnsemble(resunet, transfuse, wdff)
    h_ens, _ = train_ensemble_head(ensemble, train_loader, val_loader, device, epochs=args.ensemble_epochs)

    models = {
        "ResUNet++": resunet.to(device).eval(),
        "TransFuse": transfuse.to(device).eval(),
        "WDFFNet": wdff.to(device).eval(),
        "Ensemble": ensemble.to(device).eval(),
    }

    metrics = {name: evaluate_model(model, test_loader, device) for name, model in models.items()}

    params = {}
    fps = {}
    for name, model in models.items():
        total, trainable = parameter_stats(model)
        params[name] = total
        fps[name] = measure_fps(model, device)
        print(f"[{name}] total_params={total:,} trainable_params={trainable:,} fps={fps[name]:.2f}")

    table = build_comparison_table(metrics, params, fps)
    print("\n=== Comparison Table ===")
    print(table.to_string(index=False))
    table.to_csv("comparison_table.csv", index=False)

    plot_training_curves(h_res, "ResUNet++")
    plot_training_curves(h_trf, "TransFuse")
    plot_training_curves(h_wdf, "WDFFNet")
    plot_training_curves(h_ens, "Ensemble")

    visualize_predictions(models, test_loader, device, n_samples=3)
    plot_dice_bars(metrics)


if __name__ == "__main__":
    main()

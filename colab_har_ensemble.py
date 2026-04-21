"""
Colab-ready Hybrid Attention Ensemble (HAR) pipeline for Kvasir-SEG.

This script is intentionally organized into notebook-style cells so you can paste
it into Google Colab cell-by-cell.
"""

# =========================
# Cell 1: Setup
# =========================
# !pip install -q timm albumentations pandas tabulate gdown

import os
import random
import subprocess
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from glob import glob
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

import albumentations as A
import timm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
print(f"Device: {DEVICE} | AMP: {USE_AMP}")


# =========================
# Cell 2: Repo clone (Colab)
# =========================

def clone_repo_if_needed(repo_url: str, target_dir: str) -> None:
    if not os.path.exists(target_dir):
        subprocess.run(["git", "clone", repo_url, target_dir], check=True)
    else:
        print(f"Repo already exists: {target_dir}")


# Example for Colab:
# REPO_DIR = "/content/ensembleArchitectureBalkees"
# clone_repo_if_needed("https://github.com/abdulkareem/ensembleArchitectureBalkees.git", REPO_DIR)


# =========================
# Cell 3: Exact model architectures (clean Python classes)
# =========================

class ObjectAwareAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        mid = max(1, channels // 8)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, coarse: Optional[torch.Tensor] = None) -> torch.Tensor:
        ch = self.channel_att(feat) * feat
        if coarse is None:
            max_pool = torch.max(feat, dim=1, keepdim=True)[0]
            avg_pool = torch.mean(feat, dim=1, keepdim=True)
            sp = self.spatial_conv(torch.cat([max_pool, avg_pool], dim=1))
        else:
            sp = F.interpolate(coarse, size=feat.shape[2:], mode="bilinear", align_corners=False)
        return feat * ch * sp


class WeightedFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_a = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_b = nn.Conv2d(channels, channels, kernel_size=1)
        self.weight_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_proj = self.conv_a(a)
        b_proj = self.conv_b(b)
        w = self.weight_conv(torch.cat([a_proj, b_proj], dim=1))
        return a_proj * w + b_proj * (1.0 - w)


class WDFFNet(nn.Module):
    """Architecture kept aligned with training notebook implementation."""

    def __init__(self, pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        self.back_a = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4)
        )
        self.back_b = timm.create_model(
            "resnet50", pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4)
        )

        ch_a = self.back_a.feature_info.channels()
        ch_b = self.back_b.feature_info.channels()
        target_ch = [64, 96, 128, 160]

        self.proj_a = nn.ModuleList([nn.Conv2d(ca, tc, 1) for ca, tc in zip(ch_a, target_ch)])
        self.proj_b = nn.ModuleList([nn.Conv2d(cb, tc, 1) for cb, tc in zip(ch_b, target_ch)])
        self.fusions = nn.ModuleList([WeightedFusion(tc) for tc in target_ch])
        self.oams = nn.ModuleList([ObjectAwareAttention(tc) for tc in target_ch])

        self.up3 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 96, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(96, 64, 2, stride=2)

        self.dec_conv3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.dec_conv2 = nn.Sequential(nn.Conv2d(192, 96, 3, padding=1), nn.ReLU(inplace=True))
        self.dec_conv1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._printed_debug = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats_a = self.back_a(x)
        feats_b = self.back_b(x)
        if not self._printed_debug:
            print("[WDFFNet] Backbone A feature channels:", [f.shape[1] for f in feats_a])
            print("[WDFFNet] Backbone B feature channels:", [f.shape[1] for f in feats_b])
            self._printed_debug = True
        fused_feats = []
        for pa, pb, fa, fb, fus, oam in zip(self.proj_a, self.proj_b, feats_a, feats_b, self.fusions, self.oams):
            f = fus(pa(fa), pb(fb))
            f = oam(f, None)
            fused_feats.append(f)

        f1, f2, f3, f4 = fused_feats
        cat3 = torch.cat([self.up3(f4), f3], dim=1)
        d3 = self.dec_conv3(cat3)
        cat2 = torch.cat([self.up2(d3), f2], dim=1)
        d2 = self.dec_conv2(cat2)
        cat1 = torch.cat([self.up1(d2), f1], dim=1)
        if cat3.shape[1] != 256 or cat2.shape[1] != 192 or cat1.shape[1] != 128:
            raise RuntimeError(
                f"WDFF decoder channel mismatch: cat3={cat3.shape}, cat2={cat2.shape}, cat1={cat1.shape}"
            )
        d1 = self.dec_conv1(cat1)
        return torch.sigmoid(self.out_conv(d1))


class BiFusionBlock(nn.Module):
    def __init__(self, cnn_ch: int, trans_ch: int, out_ch: int):
        super().__init__()
        self.conv_cnn = nn.Conv2d(cnn_ch, out_ch, kernel_size=1)
        self.conv_trans = nn.Conv2d(trans_ch, out_ch, kernel_size=1)
        self.conv_out = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, cnn_feat: torch.Tensor, trans_feat: torch.Tensor) -> torch.Tensor:
        if trans_feat.ndim == 4 and trans_feat.shape[1] < 10:
            trans_feat = trans_feat.permute(0, 3, 1, 2)
        x = self.conv_cnn(cnn_feat) + self.conv_trans(trans_feat)
        return self.act(self.conv_out(x))


class TransFuseSimple(nn.Module):
    """Architecture kept aligned with training notebook implementation."""

    def __init__(self, num_classes: int = 1, pretrained: bool = True, transfuse_input_size: int = 224):
        super().__init__()
        self.transfuse_input_size = transfuse_input_size
        self.cnn = timm.create_model("efficientnet_b0", pretrained=pretrained, features_only=True)
        self.trans = timm.create_model("mobilenetv3_large_100", pretrained=pretrained, features_only=True)

        cnn_ch = self.cnn.feature_info.channels()
        trans_ch = self.trans.feature_info.channels()

        self.fuse3 = BiFusionBlock(cnn_ch[-1], trans_ch[-1], 256)
        self.fuse2 = BiFusionBlock(cnn_ch[-2], trans_ch[-2], 128)
        self.fuse1 = BiFusionBlock(cnn_ch[-3], trans_ch[-3], 64)
        self.conv_final = nn.Conv2d(64 + 128 + 256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        native_size = x.shape[2:]
        x_224 = F.interpolate(
            x, size=(self.transfuse_input_size, self.transfuse_input_size), mode="bilinear", align_corners=False
        )
        cnn_feats = self.cnn(x_224)
        trans_feats = []
        for t in self.trans(x_224):
            if t.ndim == 4 and t.shape[-1] > t.shape[1]:
                t = t.permute(0, 3, 1, 2)
            trans_feats.append(t)

        f3 = self.fuse3(cnn_feats[-1], trans_feats[-1])
        f2 = self.fuse2(cnn_feats[-2], trans_feats[-2])
        f1 = self.fuse1(cnn_feats[-3], trans_feats[-3])

        f3_up = F.interpolate(f3, size=f1.shape[2:], mode="bilinear", align_corners=False)
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode="bilinear", align_corners=False)
        out = self.conv_final(torch.cat([f3_up, f2_up, f1], dim=1))
        out = torch.sigmoid(out)
        return F.interpolate(out, size=native_size, mode="bilinear", align_corners=False)


def import_resunetplusplus_builder(resunet_repo_dir: str):
    """Imports exact ResUNet++ builder from DebeshJha repository file."""
    path = os.path.join(resunet_repo_dir, "resunet++_pytorch.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ResUNet++ file not found: {path}")
    spec = importlib.util.spec_from_file_location("resunetplusplus", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.build_resunetplusplus


# =========================
# Cell 4: Robust weight loader
# =========================

def _canonical_key(key: str) -> str:
    """Canonicalize parameter names to improve checkpoint/model key matching."""
    return key.replace("_", "").lower()


def load_partial_state_dict(model: nn.Module, ckpt_path: str, device: torch.device) -> Dict[str, int]:
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[WARN] Missing checkpoint: {ckpt_path}")
        return {"loaded": 0, "skipped": len(model.state_dict())}

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    model_dict = model.state_dict()
    clean_ckpt = {}
    for k, v in checkpoint.items():
        nk = k[7:] if k.startswith("module.") else k
        clean_ckpt[nk] = v

    loaded, skipped = [], []
    direct_hits = set()
    for k, v in clean_ckpt.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            loaded.append(k)
            direct_hits.add(k)
        else:
            skipped.append(k)

    # Fuzzy fallback for known naming drifts (e.g. backA vs back_a, trans.* prefix changes).
    canon_to_model_keys: Dict[str, List[str]] = {}
    for mk in model_dict.keys():
        canon_to_model_keys.setdefault(_canonical_key(mk), []).append(mk)

    for ck, cv in clean_ckpt.items():
        if ck in direct_hits:
            continue
        candidates = canon_to_model_keys.get(_canonical_key(ck), [])
        for mk in candidates:
            if mk not in direct_hits and model_dict[mk].shape == cv.shape:
                model_dict[mk] = cv
                loaded.append(f"{ck} -> {mk}")
                direct_hits.add(mk)
                break

    model.load_state_dict(model_dict)
    missing_in_ckpt = [k for k in model_dict.keys() if k not in clean_ckpt]
    print(
        f"Loaded layers: {len(loaded)} | Skipped layers: {len(skipped)} | "
        f"Model params not found in checkpoint: {len(missing_in_ckpt)}"
    )
    if skipped:
        print("Skipped keys (first 20):", skipped[:20])
    if missing_in_ckpt:
        print("Missing-in-ckpt keys (first 20):", missing_in_ckpt[:20])
    return {"loaded": len(loaded), "skipped": len(skipped), "missing_in_ckpt": len(missing_in_ckpt)}


# =========================
# Cell 5: Dataset / DataLoader
# =========================

class KvasirDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int = 352, augment: bool = False):
        self.df = df.reset_index(drop=True)
        if augment:
            self.tf = A.Compose([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.2),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.tf = A.Compose([A.Resize(size, size), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(row["image"]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, -1)
        out = self.tf(image=img, mask=mask)
        x = out["image"].astype("float32").transpose(2, 0, 1)
        y = (out["mask"].astype("float32") / 255.0)
        y = np.clip(y, 0.0, 1.0)
        y = (y > 0.5).astype("float32").transpose(2, 0, 1)
        return torch.from_numpy(x), torch.from_numpy(y)


def make_loaders(data_dir: str, size: int = 352, batch_size: int = 8, num_workers: int = 2):
    images = sorted(glob(os.path.join(data_dir, "images", "*.jpg")))
    masks = sorted(glob(os.path.join(data_dir, "masks", "*.jpg")))
    if len(masks) == 0:
        masks = sorted(glob(os.path.join(data_dir, "masks", "*.png")))
    assert len(images) == len(masks), "Image/mask count mismatch"

    df = pd.DataFrame({"image": images, "mask": masks}).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = df.sample(frac=0.1, random_state=42)
    train_df = df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = KvasirDataset(train_df, size=size, augment=True)
    val_ds = KvasirDataset(val_df, size=size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


# =========================
# Cell 6: HAR Ensemble
# =========================

class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        max_map = torch.max(x, dim=1, keepdim=True)[0]
        avg_map = torch.mean(x, dim=1, keepdim=True)
        return x * self.spatial(torch.cat([max_map, avg_map], dim=1))


class HAREnsemble(nn.Module):
    def __init__(self, resunetpp: nn.Module, wdff: nn.Module, transfuse: nn.Module):
        super().__init__()
        self.resunetpp = resunetpp.eval()
        self.wdff = wdff.eval()
        self.transfuse = transfuse.eval()
        # Backward-compatible aliases used by some notebook cells.
        self.r = self.resunetpp
        self.w = self.wdff
        self.t = self.transfuse

        for m in [self.resunetpp, self.wdff, self.transfuse]:
            for p in m.parameters():
                p.requires_grad = False

        self.attn = ChannelSpatialAttention(in_channels=3)
        self.weight_head = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1),
        )
        self._printed_debug = False

    def _align(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return normalize_segmentation_output(pred, ref_shape=ref.shape[2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            p1 = self.resunetpp(x)
            p2 = self.wdff(x)
            p3 = self.transfuse(x)

        p1 = self._align(p1, x)
        p2 = self._align(p2, x)
        p3 = self._align(p3, x)
        if not self._printed_debug:
            print(f"[HAR] Shapes -> p1:{tuple(p1.shape)} p2:{tuple(p2.shape)} p3:{tuple(p3.shape)}")
            self._printed_debug = True

        stack = torch.cat([p1, p2, p3], dim=1)
        stack = self.attn(stack)
        logits = self.weight_head(stack)
        weights = torch.softmax(logits, dim=1)
        fused = (weights * stack).sum(dim=1, keepdim=True)
        return fused


# =========================
# Cell 7: Metrics + evaluation
# =========================



def normalize_segmentation_output(pred: torch.Tensor, ref_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Normalize model output to BCHW tensor and optionally resize to ref_shape (H, W)."""
    if isinstance(pred, (list, tuple)):
        if len(pred) == 0:
            raise ValueError("Model output is empty list/tuple.")
        pred = pred[0]

    if pred.ndim == 2:
        # HW -> BCHW
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(1)
    elif pred.ndim == 4 and pred.shape[-1] in (1, 2, 3) and pred.shape[1] not in (1, 2, 3):
        # NHWC -> NCHW (more robust condition)
        pred = pred.permute(0, 3, 1, 2)
    elif pred.ndim != 4:
        raise ValueError(f"Unsupported prediction shape: {tuple(pred.shape)}")

    # If model accidentally returns multi-channel logits, keep first channel for binary segmentation.
    if pred.shape[1] > 1:
        pred = pred[:, :1, ...]

    if ref_shape is not None and pred.shape[2:] != ref_shape:
        pred = F.interpolate(pred, size=ref_shape, mode="bilinear", align_corners=False)
    return pred

def _bin(pred: torch.Tensor, th: float = 0.5) -> torch.Tensor:
    return (pred > th).float()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = _bin(pred)
    target = _bin(target)
    pred_f = pred.reshape(-1)
    tar_f = target.reshape(-1)

    tp = ((pred_f == 1) & (tar_f == 1)).sum().item()
    fp = ((pred_f == 1) & (tar_f == 0)).sum().item()
    fn = ((pred_f == 0) & (tar_f == 1)).sum().item()
    tn = ((pred_f == 0) & (tar_f == 0)).sum().item()

    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = (2 * precision * recall + 1e-6) / (precision + recall + 1e-6)
    acc = (tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)
    return {"Dice": dice, "IoU": iou, "Accuracy": acc, "Precision": precision, "Recall": recall, "F1": f1}


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    agg = {"Dice": [], "IoU": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": []}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = normalize_segmentation_output(model(x), ref_shape=y.shape[2:])
        batch_metrics = compute_metrics(p, y)
        for k, v in batch_metrics.items():
            agg[k].append(v)
    return {k: float(np.mean(v)) for k, v in agg.items()}


# =========================
# Cell 8: Train only HAR head
# =========================

@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_f = pred.reshape(pred.shape[0], -1)
    target_f = target.reshape(target.shape[0], -1)
    inter = (pred_f * target_f).sum(dim=1)
    union = pred_f.sum(dim=1) + target_f.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def train_har_head(model: HAREnsemble, train_loader: DataLoader, cfg: TrainConfig):
    optimizer = torch.optim.Adam(
        list(model.attn.parameters()) + list(model.weight_head.parameters()),
        lr=cfg.lr,
    )
    scaler = GradScaler("cuda", enabled=USE_AMP)

    model.to(DEVICE)
    for ep in range(1, cfg.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=USE_AMP):
                p = normalize_segmentation_output(model(x), ref_shape=y.shape[2:])
                loss_dice = dice_loss(p, y)
            # BCE is not autocast-safe; run in full precision outside autocast.
            loss_bce = F.binary_cross_entropy(p.float(), y.float())
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
        print(f"Epoch {ep}/{cfg.epochs} | HAR head loss: {run_loss / len(train_loader):.4f}")


# =========================
# Cell 9: End-to-end run
# =========================

# Example paths for Google Drive checkpoints:
# resunet_ckpt = "/content/drive/MyDrive/BalkuProject_Outputs/best_resunetpp_model.pth"
# wdff_ckpt = "/content/drive/MyDrive/BalkuProject_Outputs/best_wdffnet_model.pth"  # optional
# transfuse_ckpt = "/content/drive/MyDrive/BalkuProject_Outputs/best_transfuse_model.pth"

# Example:
# RESUNET_REPO = "/content/ResUNetPlusPlus"
# clone_repo_if_needed("https://github.com/DebeshJha/ResUNetPlusPlus.git", RESUNET_REPO)
# ResUnetPPBuilder = import_resunetplusplus_builder(RESUNET_REPO)
# resunetpp = ResUnetPPBuilder().to(DEVICE)
# wdffnet = WDFFNet(pretrained=False, num_classes=1).to(DEVICE)
# transfuse = TransFuseSimple(num_classes=1, pretrained=False).to(DEVICE)
#
# load_partial_state_dict(resunetpp, resunet_ckpt, DEVICE)
# load_partial_state_dict(wdffnet, wdff_ckpt, DEVICE)        # handled gracefully if missing
# load_partial_state_dict(transfuse, transfuse_ckpt, DEVICE)
#
# train_loader, val_loader = make_loaders("/content/data/Kvasir-SEG", size=352, batch_size=8, num_workers=2)
# har = HAREnsemble(resunetpp, wdffnet, transfuse).to(DEVICE)
# train_har_head(har, train_loader, TrainConfig(epochs=12, lr=1e-3))
#
# base_wrappers = {
#     "ResUNet++": resunetpp,
#     "WDFFNet": wdffnet,
#     "TransFuse": transfuse,
#     "HAR Ensemble": har,
# }
#
# rows = []
# for name, mdl in base_wrappers.items():
#     m = evaluate_model(mdl, val_loader, DEVICE)
#     rows.append([name, m["Dice"], m["IoU"], m["Precision"], m["Recall"], m["F1"]])
#
# print(tabulate(rows, headers=["Model", "Dice", "IoU", "Precision", "Recall", "F1"], floatfmt=".4f", tablefmt="github"))


def _env_path(name: str) -> Optional[str]:
    raw = os.environ.get(name, "").strip()
    return raw if raw else None


def _must_exist(path: Optional[str], label: str) -> Optional[str]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def mount_drive_if_needed() -> None:
    """Mount Google Drive only when running inside Colab and not already mounted."""
    in_colab = importlib.util.find_spec("google.colab") is not None
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        return
    if in_colab:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")


def run_full_pipeline() -> None:
    """
    Execute full HAR pipeline using environment variables.

    Required:
      DATA_DIR, RESUNET_REPO, RESUNET_CKPT, TRANSFUSE_CKPT
    Optional:
      WDFF_CKPT, EPOCHS, BATCH_SIZE, IMG_SIZE, NUM_WORKERS
    """
    mount_drive_if_needed()

    data_dir = _must_exist(_env_path("DATA_DIR"), "DATA_DIR")
    resunet_repo = _must_exist(_env_path("RESUNET_REPO"), "RESUNET_REPO")
    resunet_ckpt = _must_exist(_env_path("RESUNET_CKPT"), "RESUNET_CKPT")
    transfuse_ckpt = _must_exist(_env_path("TRANSFUSE_CKPT"), "TRANSFUSE_CKPT")
    wdff_ckpt = _env_path("WDFF_CKPT")
    if wdff_ckpt and not os.path.exists(wdff_ckpt):
        print(f"[WARN] WDFF_CKPT not found: {wdff_ckpt}. WDFFNet will run with random weights.")
        wdff_ckpt = None

    missing = [
        name
        for name, value in [
            ("DATA_DIR", data_dir),
            ("RESUNET_REPO", resunet_repo),
            ("RESUNET_CKPT", resunet_ckpt),
            ("TRANSFUSE_CKPT", transfuse_ckpt),
        ]
        if value is None
    ]
    if missing:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ".\nSet them before running `%run colab_har_ensemble.py`."
        )

    epochs = int(os.environ.get("EPOCHS", "12"))
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    img_size = int(os.environ.get("IMG_SIZE", "352"))
    num_workers = int(os.environ.get("NUM_WORKERS", "2"))

    ResUnetPPBuilder = import_resunetplusplus_builder(resunet_repo)
    resunetpp = ResUnetPPBuilder().to(DEVICE)
    wdffnet = WDFFNet(pretrained=False, num_classes=1).to(DEVICE)
    transfuse = TransFuseSimple(num_classes=1, pretrained=False, transfuse_input_size=224).to(DEVICE)

    load_partial_state_dict(resunetpp, resunet_ckpt, DEVICE)
    load_partial_state_dict(wdffnet, wdff_ckpt or "", DEVICE)
    load_partial_state_dict(transfuse, transfuse_ckpt, DEVICE)

    train_loader, val_loader = make_loaders(
        data_dir,
        size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    har = HAREnsemble(resunetpp, wdffnet, transfuse).to(DEVICE)
    train_har_head(har, train_loader, TrainConfig(epochs=epochs, lr=1e-3))

    base_wrappers = {
        "ResUNet++": resunetpp,
        "WDFFNet": wdffnet,
        "TransFuse": transfuse,
        "HAR Ensemble": har,
    }

    rows = []
    for name, mdl in base_wrappers.items():
        m = evaluate_model(mdl, val_loader, DEVICE)
        rows.append([name, m["Dice"], m["IoU"], m["Accuracy"], m["Precision"], m["Recall"], m["F1"]])

    print(
        tabulate(
            rows,
            headers=["Model", "Dice", "IoU", "Accuracy", "Precision", "Recall", "F1"],
            floatfmt=".4f",
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    try:
        run_full_pipeline()
    except Exception as exc:
        print(f"[INFO] End-to-end run skipped: {exc}")
        print(
            "[INFO] To run in Colab, set env vars first:\n"
            "  %env DATA_DIR=/content/data/Kvasir-SEG\n"
            "  %env RESUNET_REPO=/content/ResUNetPlusPlus\n"
            "  %env RESUNET_CKPT=/content/drive/MyDrive/.../best_resunetpp_model.pth\n"
            "  %env WDFF_CKPT=/content/drive/MyDrive/.../best_wdffnet.pth\n"
            "  %env TRANSFUSE_CKPT=/content/drive/MyDrive/.../best_transfuse_model.pth\n"
            "  %run colab_har_ensemble.py"
        )

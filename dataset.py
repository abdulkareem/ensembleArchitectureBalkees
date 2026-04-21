import os
from glob import glob
from dataclasses import dataclass
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    data_dir: str
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 2
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


class KvasirSegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = 256, augment: bool = False):
        self.df = df.reset_index(drop=True)
        tf = [A.Resize(image_size, image_size)]
        if augment:
            tf.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.2),
            ])
        tf.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.transform = A.Compose(tf)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = cv2.cvtColor(cv2.imread(row["image"]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, -1)

        out = self.transform(image=image, mask=mask)
        image = out["image"].astype("float32").transpose(2, 0, 1)
        mask = (out["mask"].astype("float32") / 255.0)
        mask = np.clip(mask, 0.0, 1.0)
        mask = (mask > 0.5).astype("float32").transpose(2, 0, 1)
        return torch.from_numpy(image), torch.from_numpy(mask)


def build_kvasir_splits(data_dir: str, seed: int = 42, val_ratio: float = 0.1, test_ratio: float = 0.1):
    images = sorted(glob(os.path.join(data_dir, "images", "*.jpg")))
    masks = sorted(glob(os.path.join(data_dir, "masks", "*.jpg")))
    if not masks:
        masks = sorted(glob(os.path.join(data_dir, "masks", "*.png")))
    assert len(images) == len(masks), "Image/mask count mismatch"

    df = pd.DataFrame({"image": images, "mask": masks})
    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=seed, shuffle=True)
    train_df, val_df = train_test_split(train_df, test_size=val_ratio / (1 - test_ratio), random_state=seed, shuffle=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_df, val_df, test_df = build_kvasir_splits(cfg.data_dir, cfg.seed, cfg.val_ratio, cfg.test_ratio)

    train_ds = KvasirSegDataset(train_df, image_size=cfg.image_size, augment=True)
    val_ds = KvasirSegDataset(val_df, image_size=cfg.image_size, augment=False)
    test_ds = KvasirSegDataset(test_df, image_size=cfg.image_size, augment=False)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader

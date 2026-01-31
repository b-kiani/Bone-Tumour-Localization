#!/usr/bin/env python
"""
Stage-1: Musculoskeletal representation pretraining on MURA
Task: Abnormal vs Normal classification
"""

import argparse
import os
import random
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm

# -----------------------------
# Utilities
# -----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# Dataset
# -----------------------------
class MuraDataset(Dataset):
    def __init__(self, csv_file, img_root, train=True):
        self.df = pd.read_csv(csv_file)
        self.root = Path(img_root)
        self.train = train

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["path"]
        label = int(row["label"])

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))

        # CLAHE
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)

        # Gamma augmentation (train only)
        if self.train:
            gamma = np.random.uniform(0.9, 1.1)
            img = np.clip((img / 255.0) ** (1.0 / gamma) * 255.0, 0, 255)

        img = np.repeat(img[..., None], 3, axis=-1)
        img = self.tf(img.astype(np.uint8))

        return img, label

# -----------------------------
# Training
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--model", default="tf_efficientnetv2_s")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    seed_everything()
    os.makedirs(args.outdir, exist_ok=True)

    dataset = MuraDataset(args.csv, args.img_root, train=True)
    labels = dataset.df["label"].values

    class_weights = 1.0 / np.bincount(labels)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True
    )

    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=1
    ).cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []

        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs = imgs.cuda()
            labels = labels.float().cuda()

            optimizer.zero_grad()
            logits = model(imgs).squeeze()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        mean_loss = float(np.mean(epoch_loss))
        history.append(mean_loss)
        print(f"Epoch {epoch+1}: loss = {mean_loss:.4f}")

    torch.save(
        model.state_dict(),
        os.path.join(args.outdir, f"stage1_{args.model}.pth")
    )

    pd.DataFrame({"loss": history}).to_csv(
        os.path.join(args.outdir, "history.csv"),
        index=False
    )

if __name__ == "__main__":
    main()

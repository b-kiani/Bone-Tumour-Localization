#!/usr/bin/env python
"""
Stage-2: Fine-tuning for primary bone tumor discrimination
"""

import argparse
import os
import random
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BTXRDDataset(Dataset):
    def __init__(self, csv_file, img_root, train=True):
        self.df = pd.read_csv(csv_file)
        self.root = img_root
        self.train = train

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(
            os.path.join(self.root, row["image"]),
            cv2.IMREAD_GRAYSCALE
        )
        img = cv2.resize(img, (224, 224))

        img = cv2.createCLAHE(2.0, (8, 8)).apply(img)

        img = np.repeat(img[..., None], 3, axis=-1)
        img = self.tf(img.astype(np.uint8))

        return img, int(row["label"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--stage1_weights", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    seed_everything()
    os.makedirs(args.outdir, exist_ok=True)

    dataset = BTXRDDataset(args.csv, args.img_root, train=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

    model = timm.create_model(
        "tf_efficientnetv2_s",
        pretrained=False,
        num_classes=1
    )
    model.load_state_dict(torch.load(args.stage1_weights), strict=False)
    model.cuda()

    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    criterion = nn.BCEWithLogitsLoss()

    history = []

    for epoch in range(10):
        model.train()
        preds, gts = [], []

        for imgs, labels in loader:
            imgs = imgs.cuda()
            labels = labels.float().cuda()

            optimizer.zero_grad()
            logits = model(imgs).squeeze()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            gts.extend(labels.cpu().numpy())

        auc = roc_auc_score(gts, preds)
        history.append(auc)
        print(f"Epoch {epoch+1}: AUC = {auc:.4f}")

    torch.save(
        model.state_dict(),
        os.path.join(args.outdir, "stage2_model.pth")
    )

    pd.DataFrame({"auc": history}).to_csv(
        os.path.join(args.outdir, "history.csv"),
        index=False
    )

if __name__ == "__main__":
    main()

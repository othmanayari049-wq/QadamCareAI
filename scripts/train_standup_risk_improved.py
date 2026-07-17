from pathlib import Path
import argparse
import copy
import json
import random
import re
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_MODEL = PROJECT_ROOT / "outputs" / "models" / "standup_r0_r1_r2_fusion_efficientnetb0.pth"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "standup_training" / "risk_improved"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def diabetic_patient_id(group, number):
    return f"{group}_P{((number + 1) // 2):03d}"


def extract_number(path):
    match = re.search(r"_(\d+)", Path(path).stem)
    if match is None:
        raise ValueError(f"Could not extract image number from {path}")
    return int(match.group(1))


def build_dataframe(data_root):
    data_root = Path(data_root)
    rows = []
    label_map = {"R0": 0, "R1": 1, "R2": 2}

    for group in ["R0", "R1", "R2"]:
        rgb_folder = data_root / "diabetic" / group / "RGB"
        thermal_folder = data_root / "diabetic" / group / "thermal"
        if not rgb_folder.exists():
            raise FileNotFoundError(f"Missing folder: {rgb_folder}")
        if not thermal_folder.exists():
            raise FileNotFoundError(f"Missing folder: {thermal_folder}")

        for rgb_path in sorted(rgb_folder.glob("*.png")):
            number = extract_number(rgb_path)
            thermal_path = thermal_folder / rgb_path.name
            if not thermal_path.exists():
                raise FileNotFoundError(f"Missing paired thermal image for {rgb_path}")
            rows.append({
                "rgb_path": str(rgb_path),
                "thermal_path": str(thermal_path),
                "patient_id": diabetic_patient_id(group, number),
                "label": label_map[group],
                "group": group,
            })

    return pd.DataFrame(rows)


def split_by_patient(df, seed):
    patient_table = df[["patient_id", "label"]].drop_duplicates()
    train_p, temp_p = train_test_split(
        patient_table,
        test_size=0.30,
        random_state=seed,
        stratify=patient_table["label"],
    )
    val_p, test_p = train_test_split(
        temp_p,
        test_size=0.50,
        random_state=seed,
        stratify=temp_p["label"],
    )
    train_ids = set(train_p.patient_id)
    val_ids = set(val_p.patient_id)
    test_ids = set(test_p.patient_id)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    return (
        df[df.patient_id.isin(train_ids)].copy(),
        df[df.patient_id.isin(val_ids)].copy(),
        df[df.patient_id.isin(test_ids)].copy(),
    )


class RiskDataset(Dataset):
    def __init__(self, df, transform, modality="fusion"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.modality = modality

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        return self.transform(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(int(row.label), dtype=torch.long)
        rgb = self._load(row.rgb_path)
        thermal = self._load(row.thermal_path)
        if self.modality == "rgb":
            thermal = rgb.clone()
        elif self.modality == "thermal":
            rgb = thermal.clone()
        return rgb, thermal, label


class FusionEfficientNetB0(nn.Module):
    def __init__(self, num_classes=3, dropout=0.45, unfreeze_last_blocks=2):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.rgb_model = models.efficientnet_b0(weights=weights)
        self.thermal_model = models.efficientnet_b0(weights=weights)

        rgb_features = self.rgb_model.classifier[1].in_features
        thermal_features = self.thermal_model.classifier[1].in_features
        self.rgb_model.classifier = nn.Identity()
        self.thermal_model.classifier = nn.Identity()

        for p in self.rgb_model.parameters():
            p.requires_grad = False
        for p in self.thermal_model.parameters():
            p.requires_grad = False

        # R0/R1/R2 is more subtle than DM/control, so we unfreeze the last feature blocks.
        if unfreeze_last_blocks > 0:
            for block in list(self.rgb_model.features.children())[-unfreeze_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            for block in list(self.thermal_model.features.children())[-unfreeze_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rgb_features + thermal_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, rgb, thermal):
        rgb_features = self.rgb_model(rgb)
        thermal_features = self.thermal_model(thermal)
        fused = torch.cat([rgb_features, thermal_features], dim=1)
        return self.classifier(fused)


def make_weighted_sampler(train_df):
    counts = train_df.label.value_counts().to_dict()
    weights = train_df.label.map(lambda label: 1.0 / counts[int(label)]).values
    return WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def metrics(y_true, logits):
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist(),
    }


def run_epoch(model, loader, criterion, optimizer, device, train=False):
    model.train(train)
    total_loss = 0.0
    all_y, all_logits = [], []

    for rgb, thermal, y in loader:
        rgb = rgb.to(device)
        thermal = thermal.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(rgb, thermal)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

        total_loss += loss.item() * rgb.size(0)
        all_y.extend(y.detach().cpu().numpy().tolist())
        all_logits.extend(logits.detach().cpu().numpy().tolist())

    return total_loss / len(loader.dataset), all_y, all_logits


def main():
    parser = argparse.ArgumentParser(description="Improved STANDUP R0/R1/R2 risk-pattern training.")
    parser.add_argument("--data_root", required=True, help="Path to STANDUP_Database folder")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--modality", choices=["fusion", "thermal", "rgb"], default="fusion")
    parser.add_argument("--unfreeze_last_blocks", type=int, default=2)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--output", default=str(OUTPUT_MODEL))
    args = parser.parse_args()

    set_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("modality:", args.modality)

    df = build_dataframe(args.data_root)
    print("total paired diabetic samples:", len(df))
    print("total patients/groups:", df.patient_id.nunique())
    print(df.label.value_counts().sort_index())

    train_df, val_df, test_df = split_by_patient(df, args.seed)
    print("Leakage check: PASSED")
    print("train:", len(train_df), "pairs,", train_df.patient_id.nunique(), "patients")
    print("val:", len(val_df), "pairs,", val_df.patient_id.nunique(), "patients")
    print("test:", len(test_df), "pairs,", test_df.patient_id.nunique(), "patients")

    train_df.to_csv(RESULTS_DIR / f"train_split_{args.modality}.csv", index=False)
    val_df.to_csv(RESULTS_DIR / f"val_split_{args.modality}.csv", index=False)
    test_df.to_csv(RESULTS_DIR / f"test_split_{args.modality}.csv", index=False)

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomRotation(12),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    sampler = make_weighted_sampler(train_df)
    train_loader = DataLoader(
        RiskDataset(train_df, train_tfms, modality=args.modality),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        RiskDataset(val_df, eval_tfms, modality=args.modality),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        RiskDataset(test_df, eval_tfms, modality=args.modality),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = FusionEfficientNetB0(num_classes=3, unfreeze_last_blocks=args.unfreeze_last_blocks).to(device)

    counts = train_df.label.value_counts().sort_index()
    class_weights = torch.tensor(
        [len(train_df) / (3 * counts.get(i, 1)) for i in range(3)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_f1 = -1.0
    best_state = None
    wait = 0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_y, val_logits = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        val_m = metrics(val_y, val_logits)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_m.items()},
        })

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"val acc {val_m['accuracy']:.4f} | val macro_f1 {val_m['macro_f1']:.4f}"
        )

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_path)
            wait = 0
        else:
            wait += 1

        if wait >= args.patience:
            print("early stopping")
            break

    if best_state is None:
        raise RuntimeError("No model state was saved. Check training data and validation metrics.")

    model.load_state_dict(best_state)
    test_loss, test_y, test_logits = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    test_m = metrics(test_y, test_logits)

    results = {
        "task": "risk_improved",
        "modality": args.modality,
        "split": "patient-wise 70/15/15",
        "test_loss": test_loss,
        "training_time_minutes": (time.time() - start) / 60,
        "saved_model_path": str(output_path),
        "note": "Experimental R0/R1/R2 model. Use only if validation/test performance is acceptable.",
        **test_m,
    }

    pd.DataFrame(history).to_csv(RESULTS_DIR / f"history_{args.modality}.csv", index=False)
    with open(RESULTS_DIR / f"test_metrics_{args.modality}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nFINAL TEST RESULTS")
    for k, v in results.items():
        print(k, ":", v)
    print("\nmodel saved to:", output_path)


if __name__ == "__main__":
    main()

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
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DM_OUTPUT = PROJECT_ROOT / "outputs" / "models" / "standup_rgb_thermal_fusion_efficientnetb0.pth"
RISK_OUTPUT = PROJECT_ROOT / "outputs" / "models" / "standup_r0_r1_r2_fusion_efficientnetb0.pth"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def diabetic_patient_id(group, number):
    return f"{group}_P{((number + 1) // 2):03d}"


def healthy_patient_id(number):
    if number <= 39:
        return f"H_P{number:03d}"
    return f"H_P{40 + ((number - 40) // 2):03d}"


def extract_number(path):
    match = re.search(r"_(\d+)", Path(path).stem)
    if match is None:
        raise ValueError(f"Could not extract image number from {path}")
    return int(match.group(1))


def build_dataframe(data_root, task):
    data_root = Path(data_root)
    rows = []

    for group in ["R0", "R1", "R2"]:
        rgb_folder = data_root / "diabetic" / group / "RGB"
        thermal_folder = data_root / "diabetic" / group / "thermal"
        if not rgb_folder.exists():
            raise FileNotFoundError(f"Missing folder: {rgb_folder}")

        for rgb_path in sorted(rgb_folder.glob("*.png")):
            number = extract_number(rgb_path)
            thermal_path = thermal_folder / rgb_path.name
            if not thermal_path.exists():
                raise FileNotFoundError(f"Missing paired thermal image for {rgb_path}")

            label = 1 if task == "dm_control" else {"R0": 0, "R1": 1, "R2": 2}[group]
            rows.append({
                "rgb_path": str(rgb_path),
                "thermal_path": str(thermal_path),
                "patient_id": diabetic_patient_id(group, number),
                "label": label,
                "group": group,
                "class_name": "diabetic" if task == "dm_control" else group,
            })

    if task == "dm_control":
        rgb_folder = data_root / "healthy" / "RGB"
        thermal_folder = data_root / "healthy" / "thermal"
        if not rgb_folder.exists():
            raise FileNotFoundError(f"Missing folder: {rgb_folder}")

        for rgb_path in sorted(rgb_folder.glob("*.png")):
            number = extract_number(rgb_path)
            thermal_path = thermal_folder / rgb_path.name
            if not thermal_path.exists():
                raise FileNotFoundError(f"Missing paired thermal image for {rgb_path}")
            rows.append({
                "rgb_path": str(rgb_path),
                "thermal_path": str(thermal_path),
                "patient_id": healthy_patient_id(number),
                "label": 0,
                "group": "healthy",
                "class_name": "healthy",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No images were found. Check the STANDUP_Database path.")
    return df


class FusionDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb = Image.open(row.rgb_path).convert("RGB")
        thermal = Image.open(row.thermal_path).convert("RGB")
        return self.transform(rgb), self.transform(thermal), torch.tensor(row.label, dtype=torch.long)


class FusionEfficientNetB0(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.rgb_model = models.efficientnet_b0(weights=weights)
        self.thermal_model = models.efficientnet_b0(weights=weights)
        rgb_features = self.rgb_model.classifier[1].in_features
        thermal_features = self.thermal_model.classifier[1].in_features
        self.rgb_model.classifier = nn.Identity()
        self.thermal_model.classifier = nn.Identity()

        for p in self.rgb_model.features.parameters():
            p.requires_grad = False
        for p in self.thermal_model.features.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(rgb_features + thermal_features, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_outputs),
        )

    def forward(self, rgb, thermal):
        rgb_features = self.rgb_model(rgb)
        thermal_features = self.thermal_model(thermal)
        fused = torch.cat([rgb_features, thermal_features], dim=1)
        return self.classifier(fused)


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

    train_ids, val_ids, test_ids = set(train_p.patient_id), set(val_p.patient_id), set(test_p.patient_id)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    train_df = df[df.patient_id.isin(train_ids)].copy()
    val_df = df[df.patient_id.isin(val_ids)].copy()
    test_df = df[df.patient_id.isin(test_ids)].copy()
    return train_df, val_df, test_df


def get_metrics(task, y_true, raw_output):
    y_true = np.array(y_true)
    if task == "dm_control":
        prob = torch.sigmoid(torch.tensor(raw_output)).numpy()
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        return {
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall_sensitivity": recall_score(y_true, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else 0,
            "f1_score": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, prob),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

    logits = torch.tensor(raw_output)
    probs = torch.softmax(logits, dim=1).numpy()
    pred = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist(),
    }


def run_epoch(model, loader, criterion, optimizer, device, task, train=False):
    model.train(train)
    total_loss = 0.0
    all_y = []
    all_raw = []

    for rgb, thermal, y in loader:
        rgb, thermal, y = rgb.to(device), thermal.to(device), y.to(device)
        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(rgb, thermal)
            if task == "dm_control":
                logits_for_loss = logits.squeeze(1)
                loss = criterion(logits_for_loss, y.float())
                raw = logits_for_loss.detach().cpu().numpy().tolist()
            else:
                loss = criterion(logits, y)
                raw = logits.detach().cpu().numpy().tolist()
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * rgb.size(0)
        all_y.extend(y.detach().cpu().numpy().tolist())
        all_raw.extend(raw)

    return total_loss / len(loader.dataset), all_y, all_raw


def main():
    parser = argparse.ArgumentParser(description="Train STANDUP RGB+thermal fusion EfficientNet-B0 locally.")
    parser.add_argument("--data_root", required=True, help="Path to STANDUP_Database folder")
    parser.add_argument("--task", choices=["dm_control", "risk"], required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional output .pth path")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    output_path = Path(args.output) if args.output else (DM_OUTPUT if args.task == "dm_control" else RISK_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "outputs" / "standup_training" / args.task
    results_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args.data_root, args.task)
    print("total paired samples:", len(df))
    print("total patients/groups:", df.patient_id.nunique())
    print(df.label.value_counts().sort_index())

    train_df, val_df, test_df = split_by_patient(df, args.seed)
    print("Leakage check: PASSED")
    print("train:", len(train_df), "pairs,", train_df.patient_id.nunique(), "patients")
    print("val:", len(val_df), "pairs,", val_df.patient_id.nunique(), "patients")
    print("test:", len(test_df), "pairs,", test_df.patient_id.nunique(), "patients")

    train_df.to_csv(results_dir / "train_split.csv", index=False)
    val_df.to_csv(results_dir / "val_split.csv", index=False)
    test_df.to_csv(results_dir / "test_split.csv", index=False)

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomRotation(8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(FusionDataset(train_df, train_tfms), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(FusionDataset(val_df, eval_tfms), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(FusionDataset(test_df, eval_tfms), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_outputs = 1 if args.task == "dm_control" else 3
    model = FusionEfficientNetB0(num_outputs=num_outputs).to(device)

    if args.task == "dm_control":
        diabetic_count = int((train_df.label == 1).sum())
        healthy_count = int((train_df.label == 0).sum())
        pos_weight = torch.tensor([healthy_count / diabetic_count], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        counts = train_df.label.value_counts().sort_index()
        class_weights = torch.tensor([len(train_df) / (3 * counts.get(i, 1)) for i in range(3)], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    wait = 0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, args.task, train=True)
        val_loss, val_y, val_raw = run_epoch(model, val_loader, criterion, optimizer, device, args.task, train=False)
        val_m = get_metrics(args.task, val_y, val_raw)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_m.items()}})
        print(f"epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_m['accuracy']:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_path)
            wait = 0
        else:
            wait += 1
        if wait >= args.patience:
            print("early stopping")
            break

    model.load_state_dict(best_state)
    test_loss, test_y, test_raw = run_epoch(model, test_loader, criterion, optimizer, device, args.task, train=False)
    test_m = get_metrics(args.task, test_y, test_raw)
    results = {
        "task": args.task,
        "split": "patient-wise 70/15/15",
        "test_loss": test_loss,
        "training_time_minutes": (time.time() - start) / 60,
        "saved_model_path": str(output_path),
        **test_m,
    }

    pd.DataFrame(history).to_csv(results_dir / "history.csv", index=False)
    with open(results_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nFINAL TEST RESULTS")
    for k, v in results.items():
        print(k, ":", v)
    print("\nmodel saved to:", output_path)


if __name__ == "__main__":
    main()

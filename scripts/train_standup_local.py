"""
Train STANDUP RGB + thermal models locally for QadamCare AI.

Supported tasks:
1) dm_control  : healthy/control-like vs diabetic-foot-like pattern
2) risk_r012   : R0 low-risk vs R1 medium-risk vs R2 high-risk pattern

Example commands:
python scripts/train_standup_local.py --task dm_control --data-root data/raw/STANDUP_Database --epochs 15 --batch-size 8
python scripts/train_standup_local.py --task risk_r012 --data-root data/raw/STANDUP_Database --epochs 20 --batch-size 8

Safety wording:
These models classify dataset-defined RGB/thermal foot-image patterns only.
They do not diagnose diabetes, estimate blood glucose, or predict exact future ulcer location.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def diabetic_patient_id(group: str, number: int) -> str:
    patient_number = (number + 1) // 2
    return f"{group}_P{patient_number:03d}"


def healthy_patient_id(number: int) -> str:
    if number <= 39:
        return f"H_P{number:03d}"
    patient_number = 40 + ((number - 40) // 2)
    return f"H_P{patient_number:03d}"


def _extract_number(path: Path) -> int:
    match = re.search(r"_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Could not extract image number from filename: {path.name}")
    return int(match.group(1))


def build_dm_control_dataframe(data_root: Path) -> pd.DataFrame:
    rows = []

    for group in ["R0", "R1", "R2"]:
        rgb_folder = data_root / "diabetic" / group / "RGB"
        thermal_folder = data_root / "diabetic" / group / "thermal"
        if not rgb_folder.exists():
            raise FileNotFoundError(f"Missing folder: {rgb_folder}")

        for rgb_path in sorted(rgb_folder.glob("*.png")):
            number = _extract_number(rgb_path)
            thermal_path = thermal_folder / rgb_path.name
            if not thermal_path.exists():
                raise FileNotFoundError(f"Missing thermal pair for {rgb_path}")

            rows.append({
                "rgb_path": str(rgb_path),
                "thermal_path": str(thermal_path),
                "filename": rgb_path.name,
                "patient_id": diabetic_patient_id(group, number),
                "label": 1,
                "class_name": "diabetic_foot_like",
                "group": group,
            })

    rgb_folder = data_root / "healthy" / "RGB"
    thermal_folder = data_root / "healthy" / "thermal"
    if not rgb_folder.exists():
        raise FileNotFoundError(f"Missing folder: {rgb_folder}")

    for rgb_path in sorted(rgb_folder.glob("*.png")):
        number = _extract_number(rgb_path)
        thermal_path = thermal_folder / rgb_path.name
        if not thermal_path.exists():
            raise FileNotFoundError(f"Missing thermal pair for {rgb_path}")

        rows.append({
            "rgb_path": str(rgb_path),
            "thermal_path": str(thermal_path),
            "filename": rgb_path.name,
            "patient_id": healthy_patient_id(number),
            "label": 0,
            "class_name": "healthy_control_like",
            "group": "healthy",
        })

    return pd.DataFrame(rows)


def build_risk_dataframe(data_root: Path) -> pd.DataFrame:
    rows = []
    label_map = {"R0": 0, "R1": 1, "R2": 2}

    for group, label in label_map.items():
        rgb_folder = data_root / "diabetic" / group / "RGB"
        thermal_folder = data_root / "diabetic" / group / "thermal"
        if not rgb_folder.exists():
            raise FileNotFoundError(f"Missing folder: {rgb_folder}")

        for rgb_path in sorted(rgb_folder.glob("*.png")):
            number = _extract_number(rgb_path)
            thermal_path = thermal_folder / rgb_path.name
            if not thermal_path.exists():
                raise FileNotFoundError(f"Missing thermal pair for {rgb_path}")

            rows.append({
                "rgb_path": str(rgb_path),
                "thermal_path": str(thermal_path),
                "filename": rgb_path.name,
                "patient_id": diabetic_patient_id(group, number),
                "label": label,
                "class_name": group,
                "group": group,
            })

    return pd.DataFrame(rows)


def make_patient_split(df: pd.DataFrame, seed: int):
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

    train_df = df[df.patient_id.isin(train_ids)].copy()
    val_df = df[df.patient_id.isin(val_ids)].copy()
    test_df = df[df.patient_id.isin(test_ids)].copy()

    return train_df, val_df, test_df


class FusionFootDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb = Image.open(row.rgb_path).convert("RGB")
        thermal = Image.open(row.thermal_path).convert("RGB")
        rgb = self.transform(rgb)
        thermal = self.transform(thermal)
        label = torch.tensor(row.label, dtype=torch.long)
        return rgb, thermal, label


class FusionEfficientNetB0(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.rgb_model = models.efficientnet_b0(weights=weights)
        self.thermal_model = models.efficientnet_b0(weights=weights)

        rgb_features = self.rgb_model.classifier[1].in_features
        thermal_features = self.thermal_model.classifier[1].in_features
        self.rgb_model.classifier = nn.Identity()
        self.thermal_model.classifier = nn.Identity()

        # Freeze pretrained feature extractors for a fast and stable local baseline.
        for p in self.rgb_model.features.parameters():
            p.requires_grad = False
        for p in self.thermal_model.features.parameters():
            p.requires_grad = False

        if num_classes == 1:
            self.classifier = nn.Sequential(
                nn.Dropout(0.35),
                nn.Linear(rgb_features + thermal_features, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.35),
                nn.Linear(rgb_features + thermal_features, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, num_classes),
            )

    def forward(self, rgb, thermal):
        rgb_features = self.rgb_model(rgb)
        thermal_features = self.thermal_model(thermal)
        fused = torch.cat([rgb_features, thermal_features], dim=1)
        return self.classifier(fused)


def get_transforms(image_size: int):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomRotation(8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms


def binary_metrics(y_true, probabilities):
    y_pred = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def multiclass_metrics(y_true, probabilities):
    y_pred = probabilities.argmax(axis=1)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    try:
        auc = roc_auc_score(y_true, probabilities, multi_class="ovr")
    except Exception:
        auc = None
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "roc_auc_ovr": auc,
        "confusion_matrix_labels": ["R0", "R1", "R2"],
        "confusion_matrix": cm,
        "classification_report": report,
    }


def run_epoch(model, loader, criterion, optimizer, device, task: str, train: bool):
    model.train(train)
    total_loss = 0.0
    all_y = []
    all_prob = []

    for rgb, thermal, y in loader:
        rgb = rgb.to(device)
        thermal = thermal.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(rgb, thermal)
            if task == "dm_control":
                logits = logits.squeeze(1)
                y_float = y.float()
                loss = criterion(logits, y_float)
                prob = torch.sigmoid(logits).detach().cpu().numpy()
            else:
                loss = criterion(logits, y)
                prob = torch.softmax(logits, dim=1).detach().cpu().numpy()

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * rgb.size(0)
        all_y.extend(y.detach().cpu().numpy())
        all_prob.extend(prob)

    return total_loss / len(loader.dataset), np.array(all_y), np.array(all_prob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["dm_control", "risk_r012"], required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Path to STANDUP_Database folder")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows if DataLoader gives issues")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("task:", args.task)
    print("data root:", args.data_root)

    if args.task == "dm_control":
        df = build_dm_control_dataframe(args.data_root)
        num_classes = 1
        model_filename = "standup_rgb_thermal_fusion_efficientnetb0.pth"
    else:
        df = build_risk_dataframe(args.data_root)
        num_classes = 3
        model_filename = "standup_r0_r1_r2_fusion_efficientnetb0.pth"

    print("total paired samples:", len(df))
    print("total patients/groups:", df.patient_id.nunique())
    print(df.label.value_counts().sort_index())

    train_df, val_df, test_df = make_patient_split(df, args.seed)
    print("Leakage check: PASSED")
    print("train:", len(train_df), "pairs,", train_df.patient_id.nunique(), "patients")
    print("val:", len(val_df), "pairs,", val_df.patient_id.nunique(), "patients")
    print("test:", len(test_df), "pairs,", test_df.patient_id.nunique(), "patients")

    split_dir = args.output_dir / f"standup_{args.task}_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_dir / "train_split.csv", index=False)
    val_df.to_csv(split_dir / "val_split.csv", index=False)
    test_df.to_csv(split_dir / "test_split.csv", index=False)

    train_tfms, eval_tfms = get_transforms(args.image_size)
    train_loader = DataLoader(FusionFootDataset(train_df, train_tfms), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(FusionFootDataset(val_df, eval_tfms), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(FusionFootDataset(test_df, eval_tfms), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = FusionEfficientNetB0(num_classes=num_classes).to(device)

    if args.task == "dm_control":
        diabetic_count = int((train_df.label == 1).sum())
        healthy_count = int((train_df.label == 0).sum())
        pos_weight = torch.tensor([healthy_count / max(diabetic_count, 1)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        counts = train_df.label.value_counts().sort_index()
        class_weights = torch.tensor(
            [len(train_df) / (3 * max(int(counts.get(i, 1)), 1)) for i in range(3)],
            dtype=torch.float32,
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    wait = 0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, args.task, train=True)
        val_loss, val_y, val_prob = run_epoch(model, val_loader, criterion, optimizer, device, args.task, train=False)

        if args.task == "dm_control":
            val_m = binary_metrics(val_y, val_prob)
            msg = f"val acc {val_m['accuracy']:.4f} | val f1 {val_m['f1_score']:.4f} | val auc {val_m['roc_auc']:.4f}"
        else:
            val_m = multiclass_metrics(val_y, val_prob)
            msg = f"val acc {val_m['accuracy']:.4f} | val macro-f1 {val_m['macro_f1']:.4f} | val auc {val_m['roc_auc_ovr']}"

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_m})
        print(f"epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | {msg}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, args.output_dir / model_filename)
            wait = 0
        else:
            wait += 1

        if wait >= args.patience:
            print("early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_y, test_prob = run_epoch(model, test_loader, criterion, optimizer, device, args.task, train=False)
    test_m = binary_metrics(test_y, test_prob) if args.task == "dm_control" else multiclass_metrics(test_y, test_prob)

    results = {
        "task": args.task,
        "split": "patient-wise 70/15/15",
        "test_loss": test_loss,
        "training_time_minutes": (time.time() - start) / 60,
        "model_path": str(args.output_dir / model_filename),
        "safety_note": (
            "This model classifies dataset-defined STANDUP RGB/thermal foot-image patterns only. "
            "It does not diagnose diabetes, estimate glucose level, predict exact future ulcer location, or replace clinician assessment."
        ),
        **test_m,
    }

    metrics_path = args.output_dir / f"standup_{args.task}_test_metrics.json"
    history_path = args.output_dir / f"standup_{args.task}_history.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)

    print("\nFINAL TEST RESULTS")
    for k, v in results.items():
        print(k, ":", v)
    print("\nmodel saved to:", args.output_dir / model_filename)
    print("metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()

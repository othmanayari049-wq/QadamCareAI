from pathlib import Path
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from thermal_dataset import ThermalFootDataset
from thermal_model import build_thermal_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "thermal" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "thermal"
MODEL_DIR = OUTPUT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "thermal_train.csv"
VAL_CSV = DATA_DIR / "thermal_val.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels, _, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predictions = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        auc = float("nan")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    train_dataset = ThermalFootDataset(
        TRAIN_CSV,
        image_size=IMAGE_SIZE,
        augment=True,
    )
    val_dataset = ThermalFootDataset(
        VAL_CSV,
        image_size=IMAGE_SIZE,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    train_labels = train_dataset.df["label"].values
    control_count = int((train_labels == 0).sum())
    dm_count = int((train_labels == 1).sum())

    class_weights = torch.tensor(
        [
            len(train_labels) / (2 * control_count),
            len(train_labels) / (2 * dm_count),
        ],
        dtype=torch.float32,
        device=device,
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Class weights [Control, DM]: {class_weights.tolist()}")

    model = build_thermal_model(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for images, labels, _, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_metrics['loss']:.4f} | "
            f"accuracy: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "image_size": IMAGE_SIZE,
        "class_names": ["Control Group", "DM Group"],
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "model_name": "EfficientNet-B0",
        "task": "Thermal pattern classification: DM Group vs Control Group",
        "warning": (
            "This model is not validated for diabetic-foot-ulcer detection, "
            "infection diagnosis, wound severity, or temperature measurement."
        ),
    }

    model_path = MODEL_DIR / "thermal_efficientnet_b0_best.pth"
    torch.save(checkpoint, model_path)

    history_path = OUTPUT_DIR / "thermal_training_history.json"
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print("\nTraining completed.")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Model saved to: {model_path}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()

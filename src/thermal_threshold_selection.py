from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from thermal_dataset import ThermalFootDataset
from thermal_model import build_thermal_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "thermal" / "processed"
MODEL_PATH = PROJECT_ROOT / "outputs" / "thermal" / "models" / "thermal_efficientnet_b0_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "thermal" / "threshold_selection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_CSV = DATA_DIR / "thermal_val.csv"
BATCH_SIZE = 16


def metrics_at_threshold(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = build_thermal_model(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = ThermalFootDataset(
        VAL_CSV,
        image_size=checkpoint["image_size"],
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    rows = []

    with torch.no_grad():
        for images, labels, patient_ids, filenames in loader:
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            for patient_id, filename, label, probability in zip(
                patient_ids,
                filenames,
                labels.numpy(),
                probabilities,
            ):
                rows.append(
                    {
                        "patient_id": patient_id,
                        "filename": filename,
                        "true_label": int(label),
                        "dm_probability": float(probability),
                    }
                )

    image_df = pd.DataFrame(rows)

    # Average both feet for each patient before choosing threshold.
    patient_df = (
        image_df.groupby("patient_id")
        .agg(
            true_label=("true_label", "first"),
            dm_probability=("dm_probability", "mean"),
            image_count=("filename", "count"),
        )
        .reset_index()
    )

    thresholds = np.arange(0.30, 0.91, 0.05)
    results = []

    for threshold in thresholds:
        result = metrics_at_threshold(
            patient_df["true_label"].values,
            patient_df["dm_probability"].values,
            threshold,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    # Prefer high F1. If tied, prefer higher specificity to reduce false alarms.
    best_row = results_df.sort_values(
        by=["f1", "specificity", "accuracy"],
        ascending=False,
    ).iloc[0]

    results_path = OUTPUT_DIR / "thermal_validation_thresholds.csv"
    best_path = OUTPUT_DIR / "selected_thermal_threshold.json"
    patient_path = OUTPUT_DIR / "thermal_validation_patient_probabilities.csv"

    results_df.to_csv(results_path, index=False)
    patient_df.to_csv(patient_path, index=False)

    selected = {
        "selected_threshold": float(best_row["threshold"]),
        "selection_method": (
            "Selected on validation patients using highest F1-score; "
            "specificity used as a tie-breaker."
        ),
        "validation_metrics": best_row.to_dict(),
        "warning": (
            "Threshold selection used validation data only. "
            "The held-out test set must remain untouched until final evaluation."
        ),
    }

    with open(best_path, "w", encoding="utf-8") as file:
        json.dump(selected, file, indent=2)

    print("\n=== Patient-level Validation Threshold Comparison ===\n")
    print(results_df.to_string(index=False))

    print("\n=== Selected Threshold ===")
    print(f"Threshold: {best_row['threshold']}")
    print(f"F1-score: {best_row['f1']}")
    print(f"Specificity: {best_row['specificity']}")
    print(f"Recall: {best_row['recall']}")

    print(f"\nSaved threshold comparison: {results_path}")
    print(f"Saved selected threshold: {best_path}")
    print(f"Saved validation probabilities: {patient_path}")


if __name__ == "__main__":
    main()

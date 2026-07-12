from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

from thermal_dataset import ThermalFootDataset
from thermal_model import build_thermal_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "thermal" / "processed"
MODEL_PATH = PROJECT_ROOT / "outputs" / "thermal" / "models" / "thermal_efficientnet_b0_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "thermal" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_CSV = DATA_DIR / "thermal_test.csv"
BATCH_SIZE = 16


def calculate_metrics(labels, predictions, probabilities):
    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = None

    cm = confusion_matrix(labels, predictions).tolist()

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(auc), 4) if auc is not None else None,
        "confusion_matrix": cm,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = build_thermal_model(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = ThermalFootDataset(
        TEST_CSV,
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
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            for patient_id, filename, label, probability, prediction in zip(
                patient_ids,
                filenames,
                labels.numpy(),
                probabilities,
                predictions,
            ):
                rows.append(
                    {
                        "patient_id": patient_id,
                        "filename": filename,
                        "true_label": int(label),
                        "dm_probability": float(probability),
                        "predicted_label": int(prediction),
                    }
                )

    image_df = pd.DataFrame(rows)

    image_metrics = calculate_metrics(
        image_df["true_label"],
        image_df["predicted_label"],
        image_df["dm_probability"],
    )

    patient_df = (
        image_df.groupby("patient_id")
        .agg(
            true_label=("true_label", "first"),
            dm_probability=("dm_probability", "mean"),
            image_count=("filename", "count"),
        )
        .reset_index()
    )

    patient_df["predicted_label"] = (
        patient_df["dm_probability"] >= 0.5
    ).astype(int)

    patient_metrics = calculate_metrics(
        patient_df["true_label"],
        patient_df["predicted_label"],
        patient_df["dm_probability"],
    )

    image_path = OUTPUT_DIR / "thermal_test_predictions_image_level.csv"
    patient_path = OUTPUT_DIR / "thermal_test_predictions_patient_level.csv"
    results_path = OUTPUT_DIR / "thermal_test_metrics.json"

    image_df.to_csv(image_path, index=False)
    patient_df.to_csv(patient_path, index=False)

    results = {
        "task": "DM Group vs Control Group thermal-pattern classification",
        "warning": (
            "This evaluation does not validate diabetic-foot-ulcer detection, "
            "infection diagnosis, wound severity, or temperature measurement."
        ),
        "image_level_metrics": image_metrics,
        "patient_level_metrics": patient_metrics,
    }

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n=== Held-out Test Results ===")

    print("\nImage-level metrics:")
    for key, value in image_metrics.items():
        print(f"{key}: {value}")

    print("\nPatient-level metrics:")
    for key, value in patient_metrics.items():
        print(f"{key}: {value}")

    print(f"\nSaved image predictions: {image_path}")
    print(f"Saved patient predictions: {patient_path}")
    print(f"Saved metrics: {results_path}")


if __name__ == "__main__":
    main()

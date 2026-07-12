from pathlib import Path
import json

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
THRESHOLD_PATH = PROJECT_ROOT / "outputs" / "thermal" / "threshold_selection" / "selected_thermal_threshold.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "thermal" / "evaluation_thresholded"

TEST_CSV = DATA_DIR / "thermal_test.csv"
BATCH_SIZE = 16

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_info = json.load(file)

    threshold = float(threshold_info["selected_threshold"])

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
            dm_probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            for patient_id, filename, label, probability in zip(
                patient_ids,
                filenames,
                labels.numpy(),
                dm_probabilities,
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

    # Average all available images for each person.
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
        patient_df["dm_probability"] >= threshold
    ).astype(int)

    labels = patient_df["true_label"]
    predictions = patient_df["predicted_label"]
    probabilities = patient_df["dm_probability"]

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    auc = roc_auc_score(labels, probabilities)
    cm = confusion_matrix(labels, predictions)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    results = {
        "task": "DM Group vs Control Group thermal-pattern classification",
        "decision_threshold": threshold,
        "patient_level_metrics": {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "specificity": round(float(specificity), 4),
            "f1": round(float(f1), 4),
            "auc": round(float(auc), 4),
            "confusion_matrix": cm.tolist(),
        },
        "important_note": (
            "Threshold 0.65 was selected using validation patients only. "
            "This model classifies dataset-defined DM Group versus Control Group "
            "thermal patterns; it does not diagnose diabetes, ulcers, infection, "
            "wound severity, or temperature."
        ),
    }

    patient_path = OUTPUT_DIR / "thermal_test_patient_predictions_threshold_065.csv"
    result_path = OUTPUT_DIR / "thermal_test_metrics_threshold_065.json"

    patient_df.to_csv(patient_path, index=False)

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n=== Patient-level Test Result at Fixed Threshold ===")
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion matrix: {cm.tolist()}")

    print(f"\nSaved patient predictions: {patient_path}")
    print(f"Saved metrics: {result_path}")


if __name__ == "__main__":
    main()

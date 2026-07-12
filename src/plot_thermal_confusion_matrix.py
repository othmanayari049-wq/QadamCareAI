from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "thermal"
    / "evaluation_thresholded"
    / "thermal_test_metrics_threshold_065.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "thermal" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "thermal_patient_level_confusion_matrix.png"


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        results = json.load(file)

    cm = np.array(results["patient_level_metrics"]["confusion_matrix"])
    threshold = results["decision_threshold"]

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Predicted Control", "Predicted DM"])
    ax.set_yticklabels(["True Control", "True DM"])

    ax.set_xlabel("Model Prediction")
    ax.set_ylabel("True Dataset Group")

    ax.set_title(
        "Thermal Classifier Confusion Matrix\n"
        f"Patient-Level Test Set | Threshold = {threshold}"
    )

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                fontsize=16,
            )

    fig.colorbar(image, ax=ax, label="Number of Patients")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=220)
    plt.close()

    print(f"Confusion-matrix figure saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()

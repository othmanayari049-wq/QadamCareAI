from pathlib import Path
import pandas as pd
import torch
import cv2

from thermal_gradcam import (
    load_thermal_gradcam,
    prepare_thermal_image,
    create_gradcam_overlay,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "thermal"
    / "evaluation_thresholded"
    / "thermal_test_patient_predictions_threshold_065.csv"
)

TEST_CSV = PROJECT_ROOT / "data" / "thermal" / "processed" / "thermal_test.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "thermal" / "gradcam_examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select one DM patient that was correctly predicted by the model.
    patient_predictions = pd.read_csv(PREDICTIONS_PATH)

    correct_dm = patient_predictions[
        (patient_predictions["true_label"] == 1)
        & (patient_predictions["predicted_label"] == 1)
    ]

    if correct_dm.empty:
        raise ValueError("No correctly predicted DM-group patient was found.")

    selected_patient = correct_dm.iloc[0]["patient_id"]

    test_df = pd.read_csv(TEST_CSV)

    selected_image = test_df[
        (test_df["patient_id"] == selected_patient)
        & (test_df["label"] == 1)
    ].iloc[0]

    image_path = Path(selected_image["image_path"])

    model, gradcam, checkpoint, device = load_thermal_gradcam(device)

    image_rgb, image_tensor = prepare_thermal_image(
        image_path,
        image_size=checkpoint["image_size"],
    )

    image_tensor = image_tensor.to(device)

    result = gradcam.generate(
        image_tensor,
        class_index=1,  # Explain the DM Group output.
    )

    heatmap_color, overlay = create_gradcam_overlay(
        image_rgb,
        result["heatmap"],
    )

    original_path = OUTPUT_DIR / "thermal_original.png"
    heatmap_path = OUTPUT_DIR / "thermal_model_attention_map.png"
    overlay_path = OUTPUT_DIR / "thermal_model_attention_overlay.png"

    cv2.imwrite(
        str(original_path),
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
    )

    cv2.imwrite(
        str(heatmap_path),
        cv2.cvtColor(heatmap_color, cv2.COLOR_RGB2BGR),
    )

    cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    predicted_label = (
        "DM Group pattern"
        if result["predicted_class"] == 1
        else "Control Group pattern"
    )

    print("\n=== Thermal Grad-CAM Example Created ===")
    print(f"Selected patient: {selected_patient}")
    print(f"Selected image: {image_path.name}")
    print(f"Model prediction: {predicted_label}")
    print(f"DM Group probability: {result['dm_probability']:.4f}")
    print("\nSaved files:")
    print(original_path)
    print(heatmap_path)
    print(overlay_path)

    print(
        "\nImportant: The attention map shows image regions that influenced "
        "the classifier. It does not show actual temperature hotspots or "
        "diagnose disease."
    )


if __name__ == "__main__":
    main()

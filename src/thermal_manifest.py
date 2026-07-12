from pathlib import Path
import re
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "thermal" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "thermal" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_original_image(path: Path) -> bool:
    name = path.stem.lower()

    blocked_tokens = [
        "-rotated",
        "-sharpened",
        "_rotated",
        "_sharpened",
    ]

    return not any(token in name for token in blocked_tokens)


def infer_label(path: Path):
    full_path = str(path).lower()

    if "dm group" in full_path or "\\dm\\" in full_path:
        return 1, "DM Group"

    if "control group" in full_path or "\\control\\" in full_path:
        return 0, "Control Group"

    return None, "Unknown"


def extract_patient_id(filename: str):
    match = re.match(r"^(DM|CG)(\d+)_", filename, flags=re.IGNORECASE)

    if match:
        prefix = match.group(1).upper()
        number = match.group(2).zfill(3)
        return f"{prefix}{number}"

    return None


def extract_side(filename: str):
    upper_name = filename.upper()

    if "_L" in upper_name:
        return "Left"
    if "_R" in upper_name:
        return "Right"

    return "Unknown"


def build_manifest():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Thermal raw folder not found: {RAW_DIR}")

    rows = []

    for path in RAW_DIR.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if not is_original_image(path):
            continue

        label, class_name = infer_label(path)

        if label is None:
            continue

        patient_id = extract_patient_id(path.name)

        if patient_id is None:
            continue

        rows.append(
            {
                "image_path": str(path.resolve()),
                "filename": path.name,
                "patient_id": patient_id,
                "side": extract_side(path.name),
                "label": label,
                "class_name": class_name,
            }
        )

    if not rows:
        raise ValueError(
            "No usable original thermal images were found. "
            "Check the folder names and file naming pattern."
        )

    df = pd.DataFrame(rows).sort_values(
        by=["class_name", "patient_id", "side", "filename"]
    )

    manifest_path = OUTPUT_DIR / "thermal_manifest.csv"
    df.to_csv(manifest_path, index=False)

    patient_summary = (
        df.groupby(["patient_id", "class_name", "label"])
        .size()
        .reset_index(name="image_count")
        .sort_values(["class_name", "patient_id"])
    )

    patient_summary_path = OUTPUT_DIR / "thermal_patient_summary.csv"
    patient_summary.to_csv(patient_summary_path, index=False)

    print("\nThermal manifest created successfully.\n")
    print(f"Usable original images: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}\n")

    print("Images by class:")
    print(df["class_name"].value_counts().to_string())

    print("\nUnique patients by class:")
    print(
        patient_summary.groupby("class_name")["patient_id"]
        .nunique()
        .to_string()
    )

    print(f"\nManifest saved to:\n{manifest_path}")
    print(f"\nPatient summary saved to:\n{patient_summary_path}")

    return df


if __name__ == "__main__":
    build_manifest()

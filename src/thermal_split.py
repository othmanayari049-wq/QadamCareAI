from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "thermal" / "processed"

MANIFEST_PATH = PROCESSED_DIR / "thermal_manifest.csv"
TRAIN_PATH = PROCESSED_DIR / "thermal_train.csv"
VAL_PATH = PROCESSED_DIR / "thermal_val.csv"
TEST_PATH = PROCESSED_DIR / "thermal_test.csv"


def split_patients():
    df = pd.read_csv(MANIFEST_PATH)

    patient_labels = (
        df[["patient_id", "label", "class_name"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    train_patients, temp_patients = train_test_split(
        patient_labels,
        test_size=0.30,
        stratify=patient_labels["label"],
        random_state=42,
    )

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        stratify=temp_patients["label"],
        random_state=42,
    )

    train_ids = set(train_patients["patient_id"])
    val_ids = set(val_patients["patient_id"])
    test_ids = set(test_patients["patient_id"])

    train_df = df[df["patient_id"].isin(train_ids)].copy()
    val_df = df[df["patient_id"].isin(val_ids)].copy()
    test_df = df[df["patient_id"].isin(test_ids)].copy()

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("\nPatient-safe thermal split created.\n")

    for name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(f"{name}:")
        print(f"  Images: {len(split_df)}")
        print(f"  Patients: {split_df['patient_id'].nunique()}")
        print(f"  DM images: {(split_df['label'] == 1).sum()}")
        print(f"  Control images: {(split_df['label'] == 0).sum()}")
        print()

    print("Files saved:")
    print(TRAIN_PATH)
    print(VAL_PATH)
    print(TEST_PATH)


if __name__ == "__main__":
    split_patients()

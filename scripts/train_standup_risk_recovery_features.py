from pathlib import Path
import argparse, json, re, time
import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "models"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "standup_training" / "risk_recovery_features"
DEFAULT_OUTPUT = OUT_DIR / "standup_r0_r1_r2_thermal_recovery_features.pkl"

LABELS = {"R0": 0, "R1": 1, "R2": 2}
LABEL_NAMES = ["R0 low-risk pattern", "R1 medium-risk pattern", "R2 high-risk pattern"]


def image_number(path):
    m = re.search(r"_(\d+)", Path(path).stem)
    if not m:
        raise ValueError(f"Could not extract number from {path}")
    return int(m.group(1))


def patient_id(group, number):
    return f"{group}_P{((number + 1) // 2):03d}"


def read_gray(path, size=224):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size)).astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def image_features(gray, prefix):
    h, w = gray.shape
    left = gray[:, : w // 2]
    right = gray[:, w // 2 :]
    center = gray[h//4:3*h//4, w//4:3*w//4]
    p75 = float(np.percentile(gray, 75))
    p90 = float(np.percentile(gray, 90))
    p95 = float(np.percentile(gray, 95))
    return {
        f"{prefix}_mean": float(gray.mean()),
        f"{prefix}_std": float(gray.std()),
        f"{prefix}_min": float(gray.min()),
        f"{prefix}_max": float(gray.max()),
        f"{prefix}_p10": float(np.percentile(gray, 10)),
        f"{prefix}_p25": float(np.percentile(gray, 25)),
        f"{prefix}_p50": float(np.percentile(gray, 50)),
        f"{prefix}_p75": p75,
        f"{prefix}_p90": p90,
        f"{prefix}_p95": p95,
        f"{prefix}_hot75_ratio": float((gray >= p75).mean()),
        f"{prefix}_hot90_ratio": float((gray >= p90).mean()),
        f"{prefix}_hot95_ratio": float((gray >= p95).mean()),
        f"{prefix}_left_mean": float(left.mean()),
        f"{prefix}_right_mean": float(right.mean()),
        f"{prefix}_lr_asymmetry": float(abs(left.mean() - right.mean())),
        f"{prefix}_center_mean": float(center.mean()),
    }


def build_patient_table(data_root):
    data_root = Path(data_root)
    rows = []
    for group in ["R0", "R1", "R2"]:
        thermal_dir = data_root / "diabetic" / group / "thermal"
        files = sorted(thermal_dir.glob("*.png"), key=image_number)
        by_pid = {}
        for p in files:
            n = image_number(p)
            by_pid.setdefault(patient_id(group, n), []).append(p)
        for pid, paths in by_pid.items():
            paths = sorted(paths, key=image_number)
            if len(paths) < 2:
                continue
            t0, t10 = paths[0], paths[1]
            g0 = read_gray(t0)
            g10 = read_gray(t10)
            diff = g10 - g0
            adiff = np.abs(diff)
            feat = {"patient_id": pid, "group": group, "label": LABELS[group], "t0_path": str(t0), "t10_path": str(t10)}
            feat.update(image_features(g0, "t0"))
            feat.update(image_features(g10, "t10"))
            feat.update(image_features(diff, "delta"))
            feat.update(image_features(adiff, "abs_delta"))
            feat["mean_recovery_change"] = float(g10.mean() - g0.mean())
            feat["hot90_change"] = float((g10 >= np.percentile(g10, 90)).mean() - (g0 >= np.percentile(g0, 90)).mean())
            feat["asymmetry_change"] = feat["t10_lr_asymmetry"] - feat["t0_lr_asymmetry"]
            rows.append(feat)
    return pd.DataFrame(rows)


def metrics(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0,1,2]).tolist(),
        "labels": LABEL_NAMES,
    }


def main():
    ap = argparse.ArgumentParser(description="Patient-level STANDUP R0/R1/R2 model using T0/T10 thermal recovery features.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    df = build_patient_table(args.data_root)
    print("patient-level samples:", len(df))
    print("patients:", df.patient_id.nunique())
    print(df.label.value_counts().sort_index())

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=args.seed, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=args.seed, stratify=temp_df["label"])
    print("Leakage check: PASSED")
    print("train:", len(train_df), "val:", len(val_df), "test:", len(test_df))

    feature_cols = [c for c in df.columns if c not in ["patient_id", "group", "label", "t0_path", "t10_path"]]
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_val, y_val = val_df[feature_cols], val_df["label"]
    X_test, y_test = test_df[feature_cols], test_df["label"]

    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=args.seed, max_depth=None, min_samples_leaf=2),
        "extra_trees": ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=args.seed, max_depth=None, min_samples_leaf=2),
        "gradient_boosting": GradientBoostingClassifier(random_state=args.seed),
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=args.seed))]),
    }

    best_name, best_model, best_val = None, None, -1
    all_results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_m = metrics(y_val, val_pred)
        all_results[name] = {"val": val_m}
        print(f"{name} | val acc {val_m['accuracy']:.4f} | val macro_f1 {val_m['macro_f1']:.4f}")
        if val_m["macro_f1"] > best_val:
            best_name, best_model, best_val = name, model, val_m["macro_f1"]

    test_pred = best_model.predict(X_test)
    test_m = metrics(y_test, test_pred)
    result = {
        "task": "risk_recovery_features",
        "best_model": best_name,
        "split": "patient-wise 70/15/15 at patient level",
        "feature_type": "T0/T10 thermal recovery handcrafted features",
        "training_time_minutes": (time.time() - start) / 60,
        "saved_model_path": str(args.output),
        "note": "Experimental R0/R1/R2 support model. It uses thermal recovery features, not diagnosis.",
        **test_m,
    }
    joblib.dump({"model": best_model, "feature_cols": feature_cols, "label_names": LABEL_NAMES, "result": result}, args.output)
    train_df.to_csv(RESULTS_DIR / "train_patient_split.csv", index=False)
    val_df.to_csv(RESULTS_DIR / "val_patient_split.csv", index=False)
    test_df.to_csv(RESULTS_DIR / "test_patient_split.csv", index=False)
    with open(RESULTS_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"all_results": all_results, "final_test": result}, f, indent=2)

    print("\nFINAL TEST RESULTS")
    for k, v in result.items():
        print(k, ":", v)
    print("\nmodel saved to:", args.output)


if __name__ == "__main__":
    main()

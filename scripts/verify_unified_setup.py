from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


REQUIRED_FILES = [
    ROOT / "app_unified.py",
    SRC / "input_modality_router.py",
    SRC / "unified_pdf_report.py",
    SRC / "standup_inference.py",
    SRC / "thermal_inference.py",
    SRC / "thermal_risk_zones.py",
    SRC / "ulcer_segmentation_inference.py",
]

OPTIONAL_MODEL_FILES = {
    "FUSeg ulcer segmentation": ROOT / "outputs" / "models" / "unet_efficientnet_b0_25epochs_best.pth",
    "STANDUP RGB + thermal fusion": ROOT / "outputs" / "models" / "standup_rgb_thermal_fusion_efficientnetb0.pth",
    "Pseudo-colour thermal-only": ROOT / "outputs" / "thermal" / "models" / "thermal_efficientnet_b0_best.pth",
    "Pseudo-colour threshold": ROOT / "outputs" / "thermal" / "threshold_selection" / "selected_thermal_threshold.json",
}

REQUIRED_PACKAGES = [
    "streamlit",
    "torch",
    "torchvision",
    "cv2",
    "reportlab",
    "PIL",
    "numpy",
]


def main():
    failures = []

    print("QadamCare AI unified setup verification")
    print("=" * 48)

    print("\nRequired project files")
    for path in REQUIRED_FILES:
        present = path.exists()
        print(f"[{'OK' if present else 'MISSING'}] {path.relative_to(ROOT)}")
        if not present:
            failures.append(f"Missing project file: {path}")

    print("\nPython packages")
    for package in REQUIRED_PACKAGES:
        present = importlib.util.find_spec(package) is not None
        print(f"[{'OK' if present else 'MISSING'}] {package}")
        if not present:
            failures.append(f"Missing Python package: {package}")

    print("\nModel availability by workflow")
    for label, path in OPTIONAL_MODEL_FILES.items():
        present = path.exists()
        print(f"[{'READY' if present else 'NOT READY'}] {label}: {path.relative_to(ROOT)}")

    print("\nWorkflow rules")
    print("- Visible ulcer analysis: close-up natural RGB foot/wound image only")
    print("- STANDUP analysis: matching plantar RGB + grayscale thermal pair")
    print("- Pseudo-colour thermal: coloured thermogram only")
    print("- R0/R1/R2 remains experimental and is excluded from the unified app")

    if failures:
        print("\nSETUP CHECK FAILED")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)

    print("\nSETUP CHECK PASSED")
    print("Run: python -m streamlit run app_unified.py")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
import tempfile

import cv2
import torch
import numpy as np
import streamlit as st
import albumentations as A
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

from model import build_model
from quality import assess_image_quality
from features import extract_lesion_features
from tracker import compare_visits


MODEL_PATH = ROOT / "outputs" / "models" / "unet_efficientnet_b0_25epochs_best.pth"


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("unet", "efficientnet-b0")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def run_analysis(image_path, previous_area=None, image_size=256):
    model, device = load_model()

    quality = assess_image_quality(image_path)

    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = A.Compose([A.Resize(image_size, image_size)])
    resized = transform(image=image_rgb)["image"]

    x = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    pred_mask = (prob > 0.5).astype(np.uint8)
    features = extract_lesion_features(pred_mask, prob_map=prob)

    overlay = resized.copy()
    overlay[pred_mask == 1] = [255, 0, 0]
    blended = cv2.addWeighted(resized, 0.65, overlay, 0.35, 0)

    confidence = float(prob[pred_mask == 1].mean()) if pred_mask.sum() > 0 else 0.0

    visit_result = None
    if previous_area is not None and previous_area > 0:
        visit_result = compare_visits(previous_area, features["total_area_pixels"])

    return resized, pred_mask, blended, quality, features, confidence, visit_result


st.set_page_config(page_title="QadamCare AI", layout="wide")

st.title("QadamCare AI")
st.subheader("AI-assisted diabetic foot screening and monitoring prototype")

st.warning(
    "Educational engineering prototype only. This system supports visual screening and documentation. "
    "It is not a medical diagnosis and must be reviewed by a qualified healthcare professional."
)

uploaded_file = st.file_uploader("Upload a foot image", type=["png", "jpg", "jpeg"])

previous_area = st.number_input(
    "Optional: previous visit wound area in pixels",
    min_value=0,
    value=0,
    step=1,
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = Path(tmp.name)

    if st.button("Analyze image"):
        with st.spinner("Running QadamCare AI analysis..."):
            image, mask, overlay, quality, features, confidence, visit_result = run_analysis(
                temp_path,
                previous_area=previous_area if previous_area > 0 else None,
            )

        st.success("Analysis completed.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image, caption="Original image", use_container_width=True)

        with col2:
            st.image(mask * 255, caption="AI predicted ulcer mask", use_container_width=True)

        with col3:
            st.image(overlay, caption="AI overlay", use_container_width=True)

        st.divider()

        st.subheader("Image Quality Assessment")
        st.write("Status:", quality["status"])
        st.json(quality["metrics"])

        if quality["blockers"]:
            st.error("Retake recommended:")
            for item in quality["blockers"]:
                st.write("-", item)

        if quality["warnings"]:
            st.warning("Warnings:")
            for item in quality["warnings"]:
                st.write("-", item)

        st.subheader("AI Segmentation Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("Detected regions", features["number_of_lesions"])
        c2.metric("Total area", f"{features['total_area_pixels']} px")
        c3.metric("Avg confidence", f"{confidence:.4f}")

        st.write("Lesion details")
        st.json(features["lesions"])

        if visit_result is not None:
            st.subheader("Visit-to-Visit Comparison")
            st.metric("Trend", visit_result["status"])
            st.write("Area change:", visit_result["change_pixels"], "pixels")
            st.write("Change percentage:", visit_result["change_percent"], "%")
            st.write(visit_result["message"])

        st.subheader("Clinician-Oriented Recommendation")
        if quality["status"] != "PASS":
            st.write("Image quality is not sufficient. Please retake the image before clinical review.")
        elif features["number_of_lesions"] > 0:
            st.write(
                "Visible ulcer-like region detected. Clinical review is recommended. "
                "The AI output should be used as screening support only."
            )
        else:
            st.write(
                "No visible ulcer-like region was detected by the model. "
                "Clinical review may still be needed depending on patient symptoms and risk factors."
            )
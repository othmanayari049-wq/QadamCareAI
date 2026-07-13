# Model Card: QadamCare AI

## Model Name

**QadamCare AI RGB Diabetic-Foot Segmentation Model**

---

## Project Summary

QadamCare AI is an educational engineering prototype designed to support diabetic-foot visual screening, follow-up monitoring, secondary complication review support, and clinician-facing documentation.

The core model performs RGB image segmentation to identify visible wound-like or ulcer-like regions in diabetic-foot images. The model output is combined with image-quality checks, clinician-entered findings, previous-visit comparison, and report-generation tools.

> **Medical safety notice:** QadamCare AI is not a diagnostic system. It does not confirm infection, ischemia, osteomyelitis, ulcer depth, Wagner grade, amputation risk, or treatment decisions. All outputs require review by a qualified healthcare professional.

---

## Intended Use

This model is intended for educational engineering research and prototype demonstration. It supports segmentation of visible wound-like or ulcer-like regions in RGB diabetic-foot images.

The model output is used for:

- visual segmentation support
- mask and overlay generation
- wound-like area estimation in pixels
- clinician-facing documentation support
- prototype review-priority support
- follow-up comparison support when previous visit data is provided

The model is not intended for autonomous clinical diagnosis.

---

## Out-of-Scope Use

This model must not be used to:

- diagnose diabetic-foot ulcer severity
- diagnose infection
- diagnose ischemia
- diagnose osteomyelitis
- determine ulcer depth
- confirm Wagner grade
- predict amputation risk
- prescribe treatment
- replace clinician judgment
- make emergency or triage decisions without clinician review

---

## Model Architecture

The RGB segmentation model uses:

- **Architecture:** U-Net
- **Backbone:** EfficientNet-B0
- **Task:** Binary segmentation
- **Input:** RGB foot image
- **Output:** Binary mask of visible wound-like / ulcer-like region
- **Input size during inference:** 256 × 256 pixels

The model predicts a probability map. A threshold is applied to generate a binary mask.

The generated mask is used to create:

- AI overlay image
- visible wound-like area estimate
- lesion count
- confidence score
- clinician-facing visual documentation

---

## Dataset

The RGB segmentation model was trained using the **Foot Ulcer Segmentation Challenge** dataset.

Local project split used during development:

| Split | Images | Labels |
|---|---:|---:|
| Training | 810 | 810 |
| Validation / held-out evaluation | 200 | 200 |
| Test images | 200 | Not used for reported mask metrics |

The repository does not include the dataset because medical image datasets may have licensing, privacy, and distribution restrictions.

---

## Internal Evaluation

The following metrics are from internal held-out validation evaluation on 200 labeled validation images.

| Metric | Value |
|---|---:|
| Mean Dice | 0.7903 |
| Mean IoU | 0.7018 |
| Mean Precision | 0.8557 |
| Mean Recall | 0.8147 |

These values represent prototype segmentation performance on the internal validation split. They do not represent clinical validation.

---

## Performance Interpretation

The internal validation results suggest that the segmentation model can identify many visible wound-like regions in the development dataset. However, the model performance may vary depending on:

- image quality
- lighting condition
- camera angle
- focus and blur
- background
- skin tone representation
- wound appearance
- annotation quality
- dataset distribution
- image capture protocol

The model should be interpreted as a prototype computer-vision component, not a clinical decision system.

---

## Optional Thermal Research Extension

QadamCare AI also includes an optional thermography research extension.

The thermal module is separate from the RGB segmentation model. It is included to demonstrate possible multimodal research direction, not validated clinical diagnosis.

Thermal module development dataset summary:

| Group | Images | Patients |
|---|---:|---:|
| DM Group | 244 | 122 |
| Control Group | 89 | 45 |
| Total | 333 | 167 |

Patient-level split used during development:

| Split | Images | Patients |
|---|---:|---:|
| Train | 232 | 116 |
| Validation | 49 | 25 |
| Test | 52 | 26 |

Internal thermal test performance at selected threshold:

| Evaluation | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Patient-level test | 0.9231 | 0.9048 | 1.0000 | 0.9500 | 0.9549 |

The thermal module predicts dataset-defined thermal-pattern groups only. It does not diagnose diabetes, infection, ischemia, tissue temperature abnormality, ulcer severity, or vascular disease.

---

## Inputs

### Required Input

- RGB foot image in `.jpg`, `.jpeg`, or `.png` format

### Optional Inputs

- previous visit area in pixels
- previous visit note
- clinician-entered findings
- thermal image for research extension
- measurement calibration value in pixels per centimetre

---

## Outputs

The model and app can produce:

- predicted binary mask
- AI overlay image
- visible wound-like region count
- predicted wound-like area in pixels
- confidence score
- image-quality result
- prototype review level
- previous-visit area comparison
- complication pathway review-support output
- clinician-facing Markdown report
- clinician-facing PDF report
- text report summary

---

## Clinical Support Components

The application includes additional rule-based support modules around the segmentation model:

| Component | Purpose |
|---|---|
| Image-quality assessment | Checks resolution, blur, brightness, and contrast |
| Feature extraction | Extracts visible region count, area, and confidence |
| Review-level estimation | Estimates prototype review priority |
| Clinical input summary | Summarizes pain, redness, swelling, warmth, discharge, fever, neuropathy, vascular disease, and probe-to-bone finding |
| Previous-visit comparison | Compares current and previous wound-like area in pixels |
| Advanced support logic | Provides structured review-support notes |
| Secondary pathway support | Highlights possible infection-review, vascular-review, delayed-healing, or bone-involvement review pathways |
| Report generation | Produces Markdown, PDF, and text reports |

These components are designed for documentation and review support only.

---

## Limitations

Important limitations include:

1. The model is not clinically validated.
2. The model only analyzes visible RGB image information.
3. It does not assess wound depth.
4. It does not diagnose infection, ischemia, osteomyelitis, neuropathy, or vascular disease.
5. It does not confirm Wagner grade.
6. It does not predict amputation risk.
7. Area is reported in pixels unless calibration is provided.
8. Image-coordinate localization is not validated anatomical localization.
9. Thermal outputs are research-support only.
10. The complication pathway module is rule-based and not clinically validated.
11. Performance may decrease on images outside the training distribution.
12. Clinical review is required for every output.

---

## Ethical and Safety Considerations

Medical AI systems require careful evaluation before clinical use. QadamCare AI includes safety wording throughout the app and reports to reduce the risk of overclaiming.

Any future clinical use would require:

- ethical approval
- clinician-supervised validation
- privacy review
- data governance review
- cybersecurity review
- regulatory assessment
- prospective clinical testing
- deployment monitoring
- documentation of failure modes

The system is designed to support documentation and review prioritization, not independent diagnosis.

---

## Data and Privacy

The GitHub repository intentionally excludes:

- patient images
- raw datasets
- generated reports
- trained model checkpoints
- private clinical data
- temporary outputs
- local virtual environment files

These exclusions help reduce privacy, licensing, and storage risks.

Large model weights should be stored separately through controlled institutional storage, GitHub Releases, Git LFS, or another approved mechanism after confirming dataset and licensing permissions.

---

## Recommended Use

Recommended use is limited to:

- academic demonstration
- engineering prototype evaluation
- computer-vision research discussion
- clinical documentation workflow exploration
- supervised research settings

---

## Not Recommended For

This project is not recommended for:

- real clinical deployment
- unsupervised patient use
- emergency triage
- treatment planning
- diagnosis confirmation
- medical decision-making without clinician review

---

## Model Status

```text
Prototype segmentation model implemented
Internal validation completed
Professional app integration completed
Clinical deployment not approved
External clinical validation not completed
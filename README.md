## Author

**Mohamed Othman Ayari**  
Computer Engineering Student  
Qatar University  

GitHub: [othmanayari049-wq](https://github.com/othmanayari049-wq)  
Email: othmanayari049@gmail.com  

Project: **QadamCare AI**  
Role: Project developer, AI system designer, computer-vision pipeline developer, and documentation/reporting workflow designer.

# QadamCare AI

**AI-assisted diabetic-foot screening, monitoring, secondary complication review support, and clinician documentation prototype**

QadamCare AI is an educational engineering prototype designed to support diabetic-foot visual screening, follow-up monitoring, and clinician-facing documentation. The system combines computer vision, image-quality assessment, clinician-entered findings, previous-visit comparison, optional thermography research support, rule-based complication pathway review, and professional report generation.

> **Medical safety notice:** QadamCare AI is not a diagnostic system. It does not confirm infection, ischemia, osteomyelitis, ulcer depth, Wagner grade, amputation risk, or treatment decisions. All outputs must be reviewed by a qualified healthcare professional.

---

## Table of Contents

- [Project Motivation](#project-motivation)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Architecture Overview](#architecture-overview)
- [Core Modules](#core-modules)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Model and Data Policy](#model-and-data-policy)
- [Medical Safety and Limitations](#medical-safety-and-limitations)
- [Future Work](#future-work)
- [Project Status](#project-status)

---

## Project Motivation

Diabetic-foot complications are a major healthcare challenge because delayed detection, poor documentation, and inconsistent follow-up may contribute to disease progression and increased clinical burden.

QadamCare AI was developed as an engineering prototype to explore how artificial intelligence can support:

- structured visual documentation of diabetic-foot cases
- early review prioritization of visible wound-like regions
- follow-up comparison between visits
- clinician-facing report generation
- safer and more organized screening workflows

The project does not replace clinical judgment. Instead, it provides structured decision-support information that can help organize the case for clinician review.

---

## Project Objectives

The main objectives of QadamCare AI are:

1. Build a computer-vision pipeline for detecting visible ulcer-like or wound-like regions from RGB foot images.
2. Provide image-quality checks before interpretation.
3. Estimate visible wound-like area in pixels.
4. Support previous-visit comparison for follow-up monitoring.
5. Integrate clinician-entered findings such as pain, redness, swelling, warmth, discharge, fever, neuropathy, vascular disease, and probe-to-bone finding.
6. Provide rule-based secondary complication pathway review support.
7. Explore thermography as a research extension.
8. Generate clinician-facing PDF and Markdown reports.
9. Integrate local AI documentation support using a local LLM/VLM workflow.
10. Maintain strong safety wording to avoid unsupported diagnosis or treatment claims.

---

## Key Features

### 1. RGB Foot Image Segmentation

The core AI module segments visible wound-like or ulcer-like regions from uploaded RGB foot images.

Outputs include:

- predicted binary mask
- visual overlay
- number of detected wound-like regions
- total predicted area in pixels
- confidence score
- prototype review level

---

### 2. Image Quality Assessment

Before using the image, the system checks basic image quality indicators:

- resolution
- blur/focus
- brightness
- contrast
- overall suitability

This helps identify images that may need to be retaken before clinical review.

---

### 3. Clinician-Entered Findings

The app includes a sidebar where clinician or user-entered findings can be added:

- pain level
- redness
- swelling
- warmth
- discharge or odor
- fever or systemic symptoms
- known or suspected neuropathy
- known or suspected vascular disease
- positive probe-to-bone finding

These findings are used for review-priority support only. They are not used to diagnose infection, ischemia, osteomyelitis, or ulcer severity.

---

### 4. Previous-Visit Monitoring

QadamCare AI supports follow-up comparison by allowing the user to enter:

- previous visit ID
- previous wound-like area in pixels
- previous visit note

The system compares the previous and current predicted wound-like area and reports the change.

This is useful for demonstrating monitoring logic, but the comparison is meaningful only when images are captured under similar conditions.

---

### 5. Secondary Complication Pathway Review Support

The system includes a rule-based review-support module that estimates which clinical pathway may require attention.

Supported pathways include:

- infection-review pathway
- vascular-review pathway
- delayed-healing pathway
- bone-involvement review pathway
- escalation priority

This module does not diagnose complications. It only highlights possible review pathways based on image-derived features, clinician-entered findings, and follow-up information.

---

### 6. Advanced Clinical Decision-Support Logic

The advanced support module provides structured outputs such as:

- infection review signal
- Wagner-style support wording
- calibration-dependent area estimation
- diabetes context note
- image-coordinate lesion localization
- explanation points
- safety cautions

The Wagner-style output is not a real Wagner grade. It is only a cautious documentation-support estimate based on visible image information.

---

### 7. Thermography Research Extension

The project includes an optional thermography module as a research extension.

The thermal module may provide:

- predicted pattern
- DM-group pattern probability
- threshold value
- attention overlay
- safety note

Thermography outputs are not temperature measurements and are not diagnostic conclusions.

---

### 8. Local AI Documentation Copilot

QadamCare AI includes local AI-assisted documentation support using Ollama.

The documentation copilot can:

- polish structured clinical summaries
- review images and reports locally
- generate clinician-facing documentation language
- preserve safety wording
- avoid unsupported diagnosis or treatment claims

This allows the project to demonstrate AI-assisted reporting without relying on paid cloud API calls.

---

### 9. Professional Report Export

The app can generate:

- Markdown clinical summary
- polished Markdown report
- local multimodal copilot review
- professional PDF clinician report
- plain text summary

The PDF report is designed to be readable and structured for clinician review.

---

## System Workflow

```text
Foot image upload
        |
        v
Image quality assessment
        |
        v
RGB segmentation model
        |
        v
Mask + overlay + area + confidence
        |
        v
Clinical inputs + previous visit data
        |
        v
Advanced review-support logic
        |
        v
Secondary complication pathway support
        |
        v
Clinical documentation engine
        |
        v
PDF / Markdown / text report export

QadamCareAI/
│
├── app_demo.py
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
│
├── src/
│   ├── advanced_clinical.py
│   ├── clinical_ai.py
│   ├── clinical_inputs.py
│   ├── clinical_report_engine.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── feature_status.py
│   ├── features.py
│   ├── fusion_engine.py
│   ├── llm_report_polisher.py
│   ├── local_ollama.py
│   ├── metrics.py
│   ├── model.py
│   ├── multimodal_clinical_copilot.py
│   ├── multimodal_plan.py
│   ├── pdf_report.py
│   ├── predict.py
│   ├── quality.py
│   ├── report.py
│   ├── risk.py
│   ├── secondary_complication_engine.py
│   ├── thermal_inference.py
│   ├── thermal_quality.py
│   ├── tracker.py
│   └── utils.py
│
├── data/
│   └── ignored by Git
│
├── outputs/
│   └── ignored by Git
│
└── notebooks/
    └── optional research notebooks

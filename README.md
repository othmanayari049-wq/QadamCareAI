# QadamCare AI

**Workflow-safe RGB and thermal foot-image analysis, monitoring support, and AI-assisted documentation prototype**

QadamCare AI is an educational engineering and research prototype developed at Qatar University. It combines computer vision, image-quality checks, workflow-specific input validation, relative thermal visualisation, clinician-entered context, local LLM/VLM documentation, and structured PDF reporting.

> [!IMPORTANT]
> QadamCare AI is **not a medical device and not a standalone diagnostic system**. It does not independently diagnose diabetes, diabetic-foot ulcer, infection, ischemia, osteomyelitis, wound depth, severity, future ulcer location, or treatment need. The software prepares image evidence and structured documentation; qualified clinicians remain responsible for diagnosis and clinical decisions.

## Why this matters in Qatar

The International Diabetes Federation estimates that **24.6% of adults in Qatar—approximately 409,300 people aged 20–79—were living with diabetes in 2024**. This is about one in four adults.

QadamCare is designed to organize the original images, image-quality results, measurements, trained-model outputs, and local LLM/VLM notes into one reviewable case. This can reduce disconnected review steps and help a clinician move through the available evidence more efficiently without replacing clinical assessment or blood testing.

Source: [International Diabetes Federation — Qatar country data](https://idf.org/our-network/regions-and-members/middle-east-and-north-africa/members/qatar/)

## Current status

| Area | Status |
|---|---|
| Visible ulcer-like segmentation | Implemented; prototype validation only |
| STANDUP paired RGB + grayscale thermal pattern model | Implemented; one patient-wise split only |
| Relative thermal monitoring-zone mapping | Implemented; relative image intensity only |
| Pseudo-colour thermal research workflow | Research code retained; disabled in the professional interface pending verified dataset/checkpoint provenance |
| R0/R1/R2 future-ulcer-risk classification | Experimental; Macro-F1 0.4120; excluded from the reliable primary system |
| Local LLM/VLM documentation | Implemented through Ollama; generated text may be wrong and requires human review |
| Clinical validation | Not completed |
| Regulatory approval | None |

## Why the workflows are separated

The application deliberately routes each image type to a compatible pipeline. The following inputs are **not interchangeable**:

1. **Visible ulcer analysis**
   - Input: close-up normal RGB foot/wound image
   - Output: ulcer-like segmentation mask, overlay, region count, and image-area measurements
   - Model family: FUSeg-derived segmentation

2. **STANDUP paired analysis**
   - Input: matching plantar RGB image and grayscale/monochrome thermal image
   - Output: control-like versus diabetic-participant-like dataset-pattern scores, plus relative thermal monitoring zones
   - Confidence meaning: the two softmax scores sum to 1 and indicate confidence in dataset resemblance, **not the probability that a patient has diabetes**
   - Model family: dual-branch EfficientNet-B0 fusion

3. **Pseudo-colour thermal research analysis**
   - Input: pseudo-coloured plantar thermogram
   - Research output: thermal-only dataset-pattern output, attention visualisation, and relative intensity measurements
   - Model family: separate thermal research model
   - Status: disabled in `app_qadamcare_pro.py` until the dataset and checkpoint provenance can be verified

The app blocks obvious wrong-input cases rather than sending them to an incompatible model.

## Main capabilities

- image-format and workflow validation
- RGB image-quality assessment
- close-up wound/ulcer-like segmentation
- STANDUP RGB–thermal dataset-pattern comparison
- labelled softmax confidence proxy for dataset resemblance
- relative high-, medium-, and low-monitoring thermal-zone maps
- hotspot ratio and left-right asymmetry measurements
- clinician-entered findings and follow-up context
- deterministic complication-review pathways
- local LLM structured case-report drafting
- local VLM visual-evidence documentation
- Markdown, standard PDF, and comprehensive AI-integrated PDF reports
- explicit separation between user-entered history, deterministic measurements, trained-model outputs, and generative AI text

## How AI supports clinician review

QadamCare follows a controlled evidence chain:

1. The image models produce the RGB mask, relative measurements, and STANDUP dataset-pattern scores.
2. The local VLM reviews the supplied images and drafts a clearly labelled visual-evidence note.
3. The local LLM organizes verified structured evidence into a clearly labelled case-report draft.
4. The clinician reviews the original evidence and AI-generated text, then confirms the interpretation using medical history, examination, and appropriate clinical tests.

The LLM and VLM **do not create new verified clinical facts and do not confirm a diagnosis**. They are documentation and review-support components.

## Interpretation rules

The project uses strict wording rules:

- A **healthy/control-like image pattern** does not prove that a person is healthy and does not exclude diabetes.
- A **diabetic-participant-like image pattern** does not diagnose diabetes or diabetic foot.
- The two softmax scores are confidence proxies for dataset resemblance; they are not calibrated disease probabilities.
- Diabetes history is entered by the user or clinician; it is not inferred as a confirmed fact from images.
- Thermal colours or pixel intensity are not calibrated temperature measurements unless raw calibrated thermal values are available.
- High-monitoring zones indicate relative image-intensity patterns for review; they do not predict exactly where a future ulcer will occur.
- LLM/VLM sections are generated documentation and can be incomplete, inconsistent, or incorrect.
- Medical confirmation still requires qualified clinical assessment and appropriate testing.

## Model summary

### FUSeg-derived RGB segmentation

Local development split:

| Split | Images | Labels used |
|---|---:|---:|
| Training | 810 | 810 |
| Validation | 200 | 200 |
| Test images | 200 | Not used for the reported mask metrics |

Internal held-out validation:

| Metric | Value |
|---|---:|
| Dice | 0.7903 |
| IoU | 0.7018 |
| Precision | 0.8557 |
| Recall | 0.8147 |

These are engineering validation results, not clinical-performance claims.

### STANDUP paired RGB + thermal fusion

- 415 paired RGB/thermal samples
- 227 patient/group identifiers in the local grouping logic
- patient-wise 70/15/15 split
- test split: 65 paired samples from 35 patient/groups
- one local split produced 100% accuracy, sensitivity, specificity, F1, and ROC-AUC
- exact internal metric value: 1.0000

The 100% result is promising on one small internal split, but it does **not** mean 100% certainty for a new patient. Grouped cross-validation, multiple seeds, dataset-bias investigation, calibration, Qatar-focused clinical testing, and external validation are still required.

### R0/R1/R2 experimental future-ulcer-risk module

We tested the R0/R1/R2 module to classify future-ulcer risk into three levels, but it achieved a **Macro-F1 of only 0.4120**. Because this performance was not reliable enough, we excluded the module from the final primary system and retained it only as a documented research extension.

The module must not be presented as detecting or predicting a future ulcer. It can be reconsidered only if later work uses suitable longitudinal outcome data and demonstrates materially stronger external validation.

See [MODEL_CARD.md](MODEL_CARD.md) for detailed model-specific limitations.

## Future clinical-development roadmap

The following items are future improvements and are **not implemented clinical capabilities**:

- clinician-led Qatar clinical validation
- calibrated screening-support confidence using larger representative cohorts
- patient-filtered RAG with dated and cited records
- secure EHR integration and longitudinal visit comparison
- 3D wound-depth and volume measurement
- clinician correction and feedback loop
- external validation, fairness testing, and regulatory assessment

The intended future role is supervised clinical decision support: AI organizes and prioritizes evidence, while the clinician remains responsible for diagnosis, treatment, and triage.

## Application entry point

Use the professional unified application:

```powershell
python -m streamlit run app_qadamcare_pro.py
```

Older files such as `app_demo.py` and `app_unified.py` remain useful for development and debugging, but `app_qadamcare_pro.py` is the recommended interface.

## Installation

### 1. Clone and enter the project

```powershell
git clone https://github.com/othmanayari049-wq/QadamCareAI.git
cd QadamCareAI
git switch fix/unified-multimodal-routing
```

### 2. Create a virtual environment

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Add local model checkpoints

Checkpoints are intentionally excluded from Git. Place approved local checkpoints under:

```text
outputs/models/
```

Main STANDUP fusion checkpoint:

```text
outputs/models/standup_rgb_thermal_fusion_efficientnetb0.pth
```

FUSeg/thermal checkpoints should follow the paths expected by their inference modules. Do not upload restricted or dataset-derived weights until redistribution rights are confirmed.

### 4. Optional local LLM/VLM setup

Install Ollama separately, then pull the configured models:

```powershell
ollama pull qwen2.5:3b
ollama pull qwen2.5vl:3b
```

For CPU-only mode on Windows:

```powershell
$env:OLLAMA_LLM_LIBRARY="cpu_avx2"
$env:CUDA_VISIBLE_DEVICES="-1"
$env:OLLAMA_CONTEXT_LENGTH="4096"
ollama serve
```

Copy `.env.example` values into your local environment as needed. Do not commit `.env` or Streamlit secrets.

## Local data layout

Datasets are not distributed in this repository. A local STANDUP layout used by the training scripts is:

```text
data/raw/STANDUP_Database/STANDUP_Database/
├── healthy/
│   ├── RGB/
│   └── thermal/
└── diabetic/
    ├── R0/
    │   ├── RGB/
    │   └── thermal/
    ├── R1/
    └── R2/
```

Obtain each dataset from its authorised source and follow its licence, consent, citation, and redistribution conditions.

## Training commands

DM/control-like STANDUP fusion:

```powershell
python scripts\train_standup_fusion.py `
  --task dm_control `
  --data_root "data\raw\STANDUP_Database\STANDUP_Database" `
  --epochs 15 `
  --batch_size 4
```

Experimental risk model:

```powershell
python scripts\train_standup_fusion.py `
  --task risk `
  --data_root "data\raw\STANDUP_Database\STANDUP_Database" `
  --epochs 15 `
  --batch_size 4
```

Do not present the R0/R1/R2 module as reliable unless later validation materially improves its performance.

## Repository structure

```text
QadamCareAI/
├── app_qadamcare_pro.py          # recommended professional UI
├── app_unified.py                # validated workflow-safe base UI
├── app_demo.py                   # older development UI
├── scripts/                      # training, verification, and utilities
├── src/                          # models, inference, routing, reports, LLM/VLM
├── outputs/                      # local generated files and weights; ignored
├── data/                         # local datasets; ignored
├── README.md
├── MODEL_CARD.md
├── SECURITY.md
├── DATA_AND_PRIVACY.md
├── DISCLAIMER.md
├── THIRD_PARTY_NOTICES.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

## Privacy and security

Do not commit:

- patient-identifiable data or real clinical records
- raw medical images without explicit authorisation
- generated reports containing names, IDs, or protected information
- restricted datasets or checkpoints
- API keys, tokens, passwords, `.env`, or `secrets.toml`
- absolute local file paths that disclose user or institution information

Read [SECURITY.md](SECURITY.md) and [DATA_AND_PRIVACY.md](DATA_AND_PRIVACY.md) before sharing, deploying, or accepting contributions.

## Licence and third-party materials

The original source code in this repository is released under the [MIT License](LICENSE).

The MIT licence does **not** automatically grant rights to third-party datasets, trained weights, publications, model downloads, logos, or other external assets. Those materials remain governed by their own terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contribution and conduct

Contributions are welcome when they preserve privacy, workflow separation, reproducibility, and non-diagnostic wording. Read:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)

## Citation

Use the metadata in [CITATION.cff](CITATION.cff) when citing the software. Dataset and paper citations must be added separately according to their original sources.

## Author

**Othman Ayari**  
Computer Engineering, Qatar University  
GitHub: [@othmanayari049-wq](https://github.com/othmanayari049-wq)  
Email: `othmanayari049@gmail.com`

## Final disclaimer

This project is for learning, engineering demonstration, and supervised research exploration. It has not undergone prospective clinical validation, regulatory review, production security assessment, or hospital deployment approval. Do not use it for emergency decisions, patient self-diagnosis, autonomous triage, or treatment planning.

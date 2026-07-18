# QadamCare AI

**Workflow-safe RGB and thermal foot-image analysis, monitoring support, and AI-assisted documentation prototype**

QadamCare AI is an educational engineering and research prototype developed at Qatar University. It combines computer vision, image-quality checks, workflow-specific input validation, relative thermal visualisation, clinician-entered context, local LLM/VLM documentation, and structured PDF reporting.

> [!IMPORTANT]
> QadamCare AI is **not a medical device and not a diagnostic system**. It does not diagnose diabetes, diabetic-foot ulcer, infection, ischemia, osteomyelitis, wound depth, severity, future ulcer location, or treatment need. Every result requires qualified clinician review.

## Current status

| Area | Status |
|---|---|
| Visible ulcer-like segmentation | Implemented; prototype validation only |
| STANDUP paired RGB + grayscale thermal pattern model | Implemented; one patient-wise split only |
| Relative thermal monitoring-zone mapping | Implemented; relative image intensity only |
| Pseudo-colour thermal research workflow | Implemented as a separate legacy/research path |
| R0/R1/R2 risk-pattern classification | Experimental and not used as a reliable primary output |
| Local LLM/VLM documentation | Implemented through Ollama; generated text may be wrong |
| Clinical validation | Not completed |
| Regulatory approval | None |

## Why the workflows are separated

The application deliberately routes each image type to a compatible pipeline. The following inputs are **not interchangeable**:

1. **Visible ulcer analysis**
   - Input: close-up normal RGB foot/wound image
   - Output: ulcer-like segmentation mask, overlay, region count, image area measurements
   - Model family: FUSeg-derived segmentation

2. **STANDUP paired analysis**
   - Input: matching plantar RGB image and grayscale/monochrome thermal image
   - Output: dataset-defined healthy/control-like vs diabetic-foot-like image pattern, plus relative thermal monitoring zones
   - Model family: dual-branch EfficientNet-B0 fusion

3. **Pseudo-colour thermal research analysis**
   - Input: pseudo-coloured plantar thermogram
   - Output: thermal-only dataset-pattern output, attention visualisation, relative intensity measurements
   - Model family: separate thermal research model

The app blocks obvious wrong-input cases rather than sending them to an incompatible model.

## Main capabilities

- image-format and workflow validation
- RGB image-quality assessment
- close-up wound/ulcer-like segmentation
- STANDUP RGB–thermal fusion classification
- relative high-, medium-, and low-monitoring thermal-zone maps
- hotspot ratio and left-right asymmetry measurements
- clinician-entered findings and follow-up context
- rule-based complication review pathways
- local LLM clinical-reasoning documentation
- local VLM visual documentation
- Markdown, standard PDF, and comprehensive AI-integrated PDF reports
- explicit separation between user-entered history, deterministic measurements, trained-model outputs, and generative AI text

## Interpretation rules

The project uses strict wording rules:

- A **healthy/control-like image pattern** does not prove that a person is healthy and does not exclude diabetes.
- A **diabetic-foot-like image pattern** does not diagnose diabetes or diabetic foot.
- Diabetes history is entered by the user or clinician; it is not inferred from images.
- Thermal colours or pixel intensity are not calibrated temperature measurements unless raw calibrated thermal values are available.
- High-monitoring zones indicate relative image-intensity patterns for review; they do not predict exactly where a future ulcer will occur.
- LLM/VLM sections are generated documentation and can be incomplete, inconsistent, or incorrect.

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

That perfect result must be interpreted cautiously because it comes from a small dataset and one split. Grouped cross-validation, multiple seeds, dataset-bias investigation, and external validation are still required.

### R0/R1/R2 experimental module

The best early three-class result remained weak (approximately 41% macro-F1). It is therefore documented as an experimental research extension and is not presented as a reliable clinical or primary product capability.

See [MODEL_CARD.md](MODEL_CARD.md) for detailed model-specific limitations.

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
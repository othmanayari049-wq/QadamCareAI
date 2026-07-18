# Model Card: QadamCare AI Multimodal Prototype

## Overview

QadamCare AI is a local educational engineering and research prototype that combines multiple independent computer-vision and documentation modules. It is not one universal medical model.

The system currently contains:

1. FUSeg-derived close-up RGB ulcer-like segmentation
2. STANDUP paired plantar RGB + grayscale thermal fusion classification
3. Relative thermal monitoring-zone visualisation
4. A separate pseudo-colour thermal research classifier
5. Rule-based review pathways
6. Local Ollama LLM/VLM documentation modules

> **Safety notice:** The system is not a medical device and has not been clinically validated. It does not diagnose diabetes, diabetic-foot ulcer, infection, ischemia, osteomyelitis, wound depth, severity, future ulcer location, or treatment need.

## Intended use

Permitted research uses include:

- supervised engineering demonstration
- workflow and user-interface evaluation
- image segmentation research
- paired RGB/thermal pattern-classification research
- relative thermal visualisation
- documentation and report-generation experiments
- clinician-supervised research discussion

Every output requires human review.

## Out-of-scope use

Do not use QadamCare AI for:

- patient self-diagnosis
- autonomous diabetes diagnosis
- emergency or clinical triage
- treatment selection
- medication, antibiotics, dressing, debridement, or surgery decisions
- ulcer-depth or severity grading
- infection, ischemia, osteomyelitis, gangrene, or sepsis confirmation
- amputation-risk or prognosis prediction
- exact future-ulcer-location prediction
- unsupervised hospital deployment

## Workflow 1: close-up RGB ulcer-like segmentation

### Input contract

- one close-up, normal RGB foot/wound image
- sufficient resolution, brightness, focus, and contrast
- no full-person scene, face-dominant image, or thermal image

### Architecture

- task: binary segmentation
- architecture: U-Net
- encoder/backbone: EfficientNet-B0
- inference size: 256 × 256
- output: probability map and thresholded visible wound-like mask

### Development dataset

FUSeg-derived local development split:

| Split | Images | Masks used |
|---|---:|---:|
| Training | 810 | 810 |
| Validation | 200 | 200 |
| Test images | 200 | Not used for the reported mask metrics |

The repository does not redistribute the dataset.

### Internal validation

| Metric | Value |
|---|---:|
| Dice | 0.7903 |
| IoU | 0.7018 |
| Precision | 0.8557 |
| Recall | 0.8147 |

These are internal held-out engineering metrics, not clinical-performance measures.

### Outputs

- visible ulcer-like mask and overlay
- detected-region count
- estimated area in pixels
- area percentage of analysed image
- dominant image zone
- confidence proxy

### Known limitations

- false positives on backgrounds, faces, skin-like objects, shadows, and out-of-distribution scenes
- no wound-depth measurement
- no tissue classification
- no infection, ischemia, or osteomyelitis diagnosis
- pixel area is not physical wound area without validated calibration
- image-coordinate zones are not validated anatomical localisation

## Workflow 2: STANDUP paired RGB + grayscale thermal fusion

### Input contract

- one plantar RGB image
- one matching grayscale/monochrome thermal image
- both images from the same participant/capture context
- pseudo-colour thermal images are not accepted by this workflow

### Architecture

- two EfficientNet-B0 branches
- one branch for RGB, one for grayscale thermal represented as three channels
- concatenated features
- binary output

### Dataset and grouping

Local paired dataset summary:

| Group | Paired samples |
|---|---:|
| Healthy/control | 125 |
| Diabetic groups combined | 290 |
| Total | 415 |

Local grouping logic represented 227 patient/group identifiers. Splitting was patient/group-wise to reduce direct leakage.

### Internal single-split evaluation

| Metric | Value |
|---|---:|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Sensitivity/recall | 1.0000 |
| Specificity | 1.0000 |
| F1 | 1.0000 |
| ROC-AUC | 1.0000 |

Test split: 65 paired samples from 35 patient/groups, with TN=21, FP=0, FN=0, TP=44.

### Required interpretation

The output is only:

- `dataset-defined healthy/control-like image pattern`, or
- `dataset-defined diabetic-foot-like image pattern`.

It is **not** a diabetes diagnosis. A healthy/control-like image pattern does not prove that a person is healthy or does not have diabetes. A diabetic-foot-like image pattern does not prove diabetes or diabetic foot.

### Major limitations

- small public research dataset
- one patient-wise split can produce unstable or optimistic performance
- possible dataset, camera, background, acquisition, or preprocessing shortcuts
- no grouped cross-validation reported yet
- no external clinical validation
- matching of real-world RGB and thermal images is not cryptographically or clinically verified
- no calibrated temperature values

Before any strong performance claim, run grouped cross-validation, multiple random seeds, patient-level aggregation, explainability/bias checks, and external testing.

## Experimental R0/R1/R2 classification

The diabetic subset contains dataset-defined R0, R1, and R2 groups. Several approaches were tested, including fusion EfficientNet-B0, thermal-only EfficientNet-B0, and handcrafted thermal-recovery features.

Best early result remained weak:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Initial RGB+thermal fusion | 0.4318 | 0.4120 |
| Improved fusion | 0.2727 | 0.2462 |
| Thermal-only deep model | 0.2955 | 0.3094 |
| Thermal-recovery features | 0.3182 | 0.3222 |

The module is experimental and should not be presented as reliable. It is excluded from the primary product claim.

## Workflow 3: pseudo-colour thermal research analysis

This is a separate thermal-only workflow for pseudo-coloured plantar thermograms. It is not interchangeable with STANDUP grayscale thermal images.

Potential outputs include:

- dataset-defined thermal pattern probability
- attention map/overlay
- relative display-intensity measurements

Limitations:

- pseudo-colour values depend on the camera/display colour map
- they are not automatically degrees Celsius
- attention maps are not anatomical or clinical ground truth
- the workflow does not diagnose diabetes, ischemia, infection, or vascular disease

## Relative monitoring-zone visualisation

Monitoring zones use relative image intensity and derived measurements such as:

- high-monitoring ratio
- medium-monitoring ratio
- left-right asymmetry score
- dominant image zone

The labels `LOW_MONITORING`, `MEDIUM_MONITORING`, and `HIGH_MONITORING` are prototype visualisation categories. They do not represent validated clinical risk classes, future-ulcer probability, or treatment priority.

## Rule-based review pathways

Clinician/user-entered findings may be combined with valid workflow outputs to generate review flags for:

- infection-related review
- vascular/perfusion review
- delayed-healing review
- bone-involvement review
- escalation priority

These are deterministic rules, not trained clinical diagnosis models. A positive probe-to-bone or other serious finding must be verified clinically.

## LLM/VLM documentation modules

Configured local models may include:

- text model: `qwen2.5:3b`
- vision-language model: `qwen2.5vl:3b`

Their role is documentation support. Generated text may hallucinate, contradict inputs, confuse absent and missing information, or overinterpret model outputs.

Safety requirements:

- structured application data is the factual source
- user-entered diabetes history must remain separate from image classification
- every generative section must be labelled as generated
- AI narratives must not prescribe treatment or establish diagnosis
- a new case must regenerate narratives rather than reuse stale text

## Data, privacy, and redistribution

The repository intentionally excludes:

- medical datasets
- patient images
- generated patient reports
- model checkpoints
- secrets and local configuration

Dataset access and checkpoint redistribution are controlled by their original licences, consent conditions, and institutional rules. The repository’s MIT licence covers only original code and documentation unless stated otherwise.

## Bias and fairness considerations

Possible performance differences may arise from:

- skin-tone representation
- age and sex distribution
- camera and thermal-device differences
- acquisition environment
- foot positioning
- wound type and stage
- comorbidities
- image background and compression
- class imbalance

No subgroup fairness analysis or clinical generalisability assessment has been completed.

## Failure modes

Known or plausible failure modes include:

- wrong workflow selected
- full-person RGB image falsely segmented
- pseudo-colour image sent to grayscale fusion model
- unmatched RGB and thermal pair
- missing or incompatible checkpoint
- poor image quality
- background intensity classified as a hotspot
- generative report contradicting validated model output
- sidebar history mistaken for image-derived information
- stale session-state narrative reused for another case

The application includes gates for several of these conditions, but the gates are not guaranteed to catch every misuse.

## Validation and deployment status

```text
Engineering prototype: implemented
Internal model evaluation: partially completed
Workflow input routing: implemented
External validation: not completed
Prospective clinical validation: not completed
Cybersecurity assessment: not completed
Regulatory review: not completed
Hospital deployment approval: none
```

## Required future work

- grouped cross-validation and multiple seeds
- external multi-centre evaluation
- calibrated thermal data where temperature claims are needed
- subgroup and bias analysis
- uncertainty and calibration evaluation
- failure-mode testing
- clinician usability study
- secure data architecture
- privacy impact assessment
- regulatory and clinical-governance review

## Contact

Othman Ayari  
Computer Engineering, Qatar University  
GitHub: `@othmanayari049-wq`  
Email: `othmanayari049@gmail.com`
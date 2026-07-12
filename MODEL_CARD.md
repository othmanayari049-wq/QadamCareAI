\# Model Card: QadamCare AI



\## Model Name



\*\*QadamCare AI RGB Diabetic-Foot Segmentation Model\*\*



\---



\## Intended Use



This model is intended for educational engineering research and prototype demonstration. It supports segmentation of visible wound-like or ulcer-like regions in RGB diabetic-foot images.



The model output is used for:



\- visual segmentation support

\- mask and overlay generation

\- wound-like area estimation in pixels

\- clinician-facing documentation support

\- prototype review-priority support



The model is not intended for autonomous clinical diagnosis.



\---



\## Out-of-Scope Use



This model must not be used to:



\- diagnose diabetic-foot ulcer severity

\- diagnose infection

\- diagnose ischemia

\- diagnose osteomyelitis

\- determine ulcer depth

\- confirm Wagner grade

\- predict amputation risk

\- prescribe treatment

\- replace clinician judgment



\---



\## Model Architecture



The RGB segmentation model uses:



\- \*\*Architecture:\*\* U-Net

\- \*\*Backbone:\*\* EfficientNet-B0

\- \*\*Task:\*\* Binary segmentation

\- \*\*Input:\*\* RGB foot image

\- \*\*Output:\*\* Binary mask of visible wound-like / ulcer-like region

\- \*\*Input size:\*\* 256 × 256 pixels during inference



The model predicts a probability map. A threshold is applied to generate a binary mask.



\---



\## Dataset



The RGB segmentation model was trained using the Foot Ulcer Segmentation Challenge dataset.



Local project split used during development:



| Split | Images | Labels |

|---|---:|---:|

| Training | 810 | 810 |

| Validation / held-out evaluation | 200 | 200 |

| Test images | 200 | Not used for reported mask metrics |



The repository does not include the dataset because medical image datasets may have licensing, privacy, and distribution restrictions.



\---



\## Internal Evaluation



The following metrics are from internal held-out validation evaluation on 200 labeled validation images.



| Metric | Value |

|---|---:|

| Mean Dice | 0.7903 |

| Mean IoU | 0.7018 |

| Mean Precision | 0.8557 |

| Mean Recall | 0.8147 |



These values represent prototype segmentation performance on the internal validation split. They do not represent clinical validation.



\---



\## Optional Thermal Research Extension



QadamCare AI also includes an optional thermography research extension.



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



The thermal module predicts dataset-defined thermal-pattern groups only. It does not diagnose diabetes, infection, ischemia, or tissue temperature abnormality.



\---



\## Inputs



Required input:



\- RGB foot image in `.jpg`, `.jpeg`, or `.png` format



Optional inputs:



\- previous visit area in pixels

\- clinician-entered findings

\- thermal image for research extension

\- measurement calibration value in pixels per centimetre



\---



\## Outputs



The model and app can produce:



\- predicted binary mask

\- AI overlay image

\- visible wound-like region count

\- predicted area in pixels

\- confidence score

\- image-quality result

\- prototype review level

\- complication pathway review-support output

\- clinician-facing report



\---



\## Limitations



Important limitations include:



1\. The model is not clinically validated.

2\. The model only analyzes visible RGB image information.

3\. It does not assess wound depth.

4\. It does not diagnose infection, ischemia, osteomyelitis, or neuropathy.

5\. Area is reported in pixels unless calibration is provided.

6\. Image-coordinate localization is not validated anatomical localization.

7\. Thermal outputs are research-support only.

8\. The complication pathway module is rule-based and not clinically validated.

9\. Clinical review is required for every output.



\---



\## Ethical and Safety Considerations



Medical AI systems require careful evaluation before clinical use. QadamCare AI includes safety wording throughout the app and reports to reduce the risk of overclaiming.



Any future clinical use would require:



\- ethical approval

\- clinician-supervised validation

\- privacy review

\- data governance review

\- cybersecurity review

\- regulatory assessment

\- prospective clinical testing



\---



\## Data and Privacy



The GitHub repository intentionally excludes:



\- patient images

\- raw datasets

\- generated reports

\- trained model checkpoints

\- private clinical data

\- temporary outputs



These exclusions help reduce privacy and licensing risk.



\---



\## Model Status



```text

Prototype segmentation model implemented

Internal validation completed

Professional app integration completed

Clinical deployment not approved

External clinical validation not completed


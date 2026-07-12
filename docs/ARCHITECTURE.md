\# QadamCare AI Architecture



QadamCare AI is a modular engineering prototype for diabetic-foot visual screening support, follow-up monitoring, secondary complication review support, and clinician-facing documentation.



> This system is for educational engineering demonstration only. It is not a diagnostic medical device.



\---



\## High-Level Workflow



```text

User / Clinician

&#x20;     |

&#x20;     v

Streamlit Dashboard

&#x20;     |

&#x20;     v

Patient Information + Clinical Findings + Image Upload

&#x20;     |

&#x20;     v

Image Quality Assessment

&#x20;     |

&#x20;     v

RGB Segmentation Model

&#x20;     |

&#x20;     v

Mask + Overlay + Area + Confidence

&#x20;     |

&#x20;     v

Clinical Support Logic

&#x20;     |

&#x20;     v

Secondary Complication Pathway Support

&#x20;     |

&#x20;     v

Clinical Documentation Engine

&#x20;     |

&#x20;     v

PDF / Markdown / Text Report Export



System Layers

QadamCare AI

│

├── User Interface Layer

│   ├── Streamlit dashboard

│   ├── patient information

│   ├── clinical findings

│   ├── previous visit data

│   └── image upload

│

├── Computer Vision Layer

│   ├── RGB preprocessing

│   ├── U-Net segmentation model

│   ├── mask generation

│   ├── overlay generation

│   └── lesion feature extraction

│

├── Clinical Support Layer

│   ├── image-quality assessment

│   ├── review-level estimation

│   ├── clinical input summary

│   ├── previous visit comparison

│   ├── advanced clinical support

│   └── secondary complication pathway support

│

├── Optional Thermal Research Layer

│   ├── thermal image validation

│   ├── thermal pattern classifier

│   └── attention overlay

│

├── Documentation Layer

│   ├── structured clinical report

│   ├── PDF report generator

│   ├── Markdown export

│   ├── text export

│   ├── local LLM report polisher

│   └── local multimodal copilot

│

└── Safety Layer

&#x20;   ├── non-diagnostic wording

&#x20;   ├── clinical review requirement

&#x20;   ├── limitation statements

&#x20;   └── data/model privacy warnings

Main Data Flow

The user enters patient and visit information.

The user uploads an RGB foot image.

The system checks image quality.

The image is resized and passed to the segmentation model.

The model predicts a visible wound-like region mask.

The system generates an overlay and extracts area, region count, and confidence.

Clinician-entered findings are summarized.

Previous visit area is compared when available.

Rule-based support modules estimate review priority and possible review pathways.

The system generates clinician-facing reports.

Main Modules

File	Role

app\_demo.py	Main Streamlit application

model.py	Builds the segmentation model

quality.py	Performs image-quality assessment

features.py	Extracts wound-like region features

risk.py	Estimates prototype review level

tracker.py	Compares current and previous visit area

clinical\_inputs.py	Summarizes clinician-entered findings

clinical\_ai.py	Generates clinical support summary

advanced\_clinical.py	Provides advanced review-support outputs

secondary\_complication\_engine.py	Supports secondary complication pathway review

fusion\_engine.py	Combines prototype evidence signals

thermal\_inference.py	Runs optional thermal-pattern inference

thermal\_quality.py	Validates thermal image suitability

clinical\_report\_engine.py	Builds structured clinical report content

pdf\_report.py	Generates professional PDF reports

llm\_report\_polisher.py	Polishes reports using local LLM

multimodal\_clinical\_copilot.py	Generates local multimodal documentation review

local\_ollama.py	Handles local Ollama text/vision requests

Safety Design



QadamCare AI avoids unsupported clinical claims.



The system does not diagnose:



infection

ischemia

osteomyelitis

ulcer depth

Wagner grade

amputation risk



All outputs are screening-support and documentation-support only.





Then save:



```text

File → Save

Close Notepad


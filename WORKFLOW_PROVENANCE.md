# QadamCare AI Workflow Provenance

This document records which image workflows are currently supported by verified project evidence and which paths remain disabled pending provenance checks.

## Enabled in the professional application

### 1. Visible ulcer-like RGB segmentation

- Input: one close-up normal RGB foot/wound image
- Model family: U-Net-style segmentation with an EfficientNet-B0 encoder
- Output: probability map, binary mask, overlay, region count, pixel area, and relative image-area measurements
- Status: enabled for engineering demonstration and internal validation only

### 2. STANDUP paired RGB + grayscale thermal fusion

- Input: one plantar RGB image plus its matching monochrome/grayscale thermal image
- Data layout: healthy and diabetic R0/R1/R2 groups each contain separate `RGB` and `thermal` folders
- Model family: two EfficientNet-B0 branches with feature concatenation
- Main output: dataset-defined healthy/control-like versus diabetic-foot-like image pattern
- Experimental output: R0 versus R1 versus R2 classification
- Status: binary image-pattern workflow enabled; R0/R1/R2 result remains experimental because performance was weak

## Disabled in the professional application

### Legacy pseudo-colour thermal-only path

The repository contains legacy thermal-only inference and attention-map code. However, the current repository evidence does not establish all of the following:

1. the exact training dataset used for the checkpoint;
2. that the training images were genuine pseudo-colour thermograms rather than monochrome thermal images stored in three channels;
3. the training script and split used to produce the checkpoint;
4. checkpoint metadata linking it to a documented dataset version;
5. reproducible evaluation results for pseudo-colour input;
6. redistribution and use permissions for the training data and checkpoint.

For these reasons, the professional application does not expose this workflow. The underlying legacy code is retained for audit and future verification, but it must not be presented as a validated project capability.

## Important image-format clarification

Calling `.convert("RGB")` on a grayscale thermal image does not create a pseudo-colour image. It only repeats the same grayscale intensity across three channels so that a pretrained RGB network can accept the tensor.

Example grayscale storage:

```text
R = 120, G = 120, B = 120
```

Example genuine coloured pixel:

```text
R = 255, G = 40, B = 0
```

These formats are not interchangeable.

## Requirements before re-enabling pseudo-colour analysis

The path may be reconsidered only after:

- locating and reviewing the original training script;
- documenting the authorised dataset source and image format;
- verifying checkpoint metadata and class definitions;
- reproducing training or evaluation results;
- testing input routing with real examples;
- documenting limitations in `MODEL_CARD.md`;
- confirming licensing and redistribution conditions;
- adding automated and local acceptance tests.

## Safety statement

All QadamCare outputs are research and documentation support. They are not diagnoses, calibrated temperature measurements, clinical risk grades, future-ulcer predictions, or treatment recommendations.
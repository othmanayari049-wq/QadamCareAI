---
name: Feature request
about: Propose a safe, testable improvement to QadamCare AI
title: "[Feature] "
labels: enhancement
assignees: ""
---

> [!NOTE]
> QadamCare AI is a non-diagnostic research prototype. Proposed features must preserve workflow separation, privacy, cautious medical wording, and clinician review.

## Feature summary

Give the feature a short name and describe it in one paragraph.

## Problem or need

What limitation, user need, research question, or engineering problem does this address?

## Proposed behaviour

Describe the expected workflow from input to output.

```text
Input
  ↓
Validation / routing
  ↓
Processing
  ↓
Output
```

## Affected area

Select all that apply:

- [ ] Professional Streamlit application
- [ ] Visible ulcer-like RGB segmentation
- [ ] STANDUP paired RGB + grayscale thermal analysis
- [ ] Pseudo-colour thermal research workflow
- [ ] Image-quality or modality validation
- [ ] Relative thermal monitoring zones
- [ ] Follow-up comparison
- [ ] Rule-based review pathways
- [ ] Local LLM documentation
- [ ] Local VLM documentation
- [ ] PDF/report generation
- [ ] Training, evaluation, or cross-validation
- [ ] Privacy or security
- [ ] Documentation or repository governance
- [ ] Other

## Intended users

Who benefits from this feature?

- [ ] Student or researcher
- [ ] Software developer
- [ ] Clinician reviewing a supervised demonstration
- [ ] Project maintainer
- [ ] Other

## Expected benefit

Explain the expected improvement in usability, reproducibility, safety, documentation, research value, or technical performance.

## Evidence and data requirements

- Does this require a new dataset?
- Is the dataset public and redistributable?
- Are consent, ethics, privacy, and licence terms known?
- Does it require longitudinal labels, calibration, clinical ground truth, or external validation?
- What baseline and metrics should be used?

Do not propose uploading restricted medical data to the repository.

## Safety and interpretation risks

Could the feature:

- imply diagnosis or treatment?
- confuse diabetes history with an image-pattern result?
- mix incompatible RGB or thermal inputs?
- turn relative thermal intensity into a temperature claim?
- imply future ulcer prediction without longitudinal labels?
- expose patient information?
- cause LLM/VLM-generated text to override validated outputs?

Explain the safeguards, blocking rules, disclaimers, and clinician-review requirements.

## Acceptance criteria

List objective conditions that would show the feature is complete.

- [ ] 
- [ ] 
- [ ] 

Include tests for incorrect inputs and failure cases, not only the successful path.

## Alternatives considered

Describe simpler or safer alternatives and why the proposed solution is preferable.

## Mockups, references, or examples

Attach only non-sensitive examples. Include papers, documentation, or diagrams when relevant.

## Additional context

Mention dependencies, expected hardware requirements, model size, performance constraints, or related issues.
---
name: Clinical safety concern
about: Report wording, routing, model-output, privacy, or workflow behaviour that could create unsafe medical interpretation
title: "[Clinical Safety] "
labels: clinical-safety
assignees: ""
---

> [!CAUTION]
> Do not include identifiable patient information, real medical records, unrestricted clinical images, secrets, or credentials. For an active security or privacy exposure, follow `SECURITY.md` and report it privately rather than posting details publicly.

## Safety concern summary

Explain the concern in plain language and state why it matters.

## Concern category

Select all that apply:

- [ ] Diagnostic overclaim
- [ ] Treatment or urgency recommendation without adequate basis
- [ ] Unsupported infection, ischemia, osteomyelitis, depth, severity, or amputation-risk claim
- [ ] Diabetes history confused with an image-pattern result
- [ ] Healthy/control-like result presented as proof of health or absence of diabetes
- [ ] Diabetic-foot-like result presented as a diagnosis
- [ ] Wrong image type routed to a model
- [ ] Invalid or low-quality image still produced a confident result
- [ ] Relative thermal intensity described as calibrated temperature
- [ ] Monitoring zone described as future-ulcer prediction
- [ ] Weak/experimental R0/R1/R2 result presented as reliable
- [ ] LLM/VLM hallucination, contradiction, or invented fact
- [ ] PDF/report content does not match validated outputs
- [ ] Privacy, data-handling, or consent concern
- [ ] Missing or unclear limitation statement
- [ ] Other

## Where it appears

- Application page or section:
- Selected workflow:
- Report or PDF section:
- File or module:
- Branch and commit:

## Source of the disputed statement

Select the source, when known:

- [ ] User-entered history
- [ ] Deterministic measurement or rule
- [ ] Trained model output
- [ ] LLM-generated text
- [ ] VLM-generated text
- [ ] PDF composition
- [ ] Unknown

## Reproduction steps

1. 
2. 
3. 

Describe the non-sensitive input type and sidebar selections. Do not upload protected health information.

## Observed unsafe or misleading output

Quote only the minimum text needed to show the problem.

```text
Paste the relevant wording here
```

## Why the output is unsafe or misleading

Explain the possible misunderstanding, affected user, and realistic consequence. Separate actual observed behaviour from hypothetical risk.

## Expected safe behaviour

Describe what the application should display, suppress, qualify, or block.

## Suggested replacement wording

Provide cautious replacement text when helpful. Preferred terms include:

- `dataset-defined image pattern`
- `visible ulcer-like region`
- `relative thermal monitoring zone`
- `user-entered diabetes history`
- `requires qualified clinician review`
- `not clinically validated`

## Severity estimate

- [ ] Critical — could plausibly cause immediate serious harm or exposes sensitive data
- [ ] High — materially misleading clinical interpretation
- [ ] Medium — confusing or incomplete safety wording
- [ ] Low — documentation clarity issue with limited safety impact

## Temporary mitigation

State whether the feature should be disabled, hidden, labelled experimental, or avoided until corrected.

## Additional context

Include related issues, references, or de-identified screenshots.
---
name: Bug report
about: Report a reproducible technical problem in QadamCare AI
title: "[Bug] "
labels: bug
assignees: ""
---

> [!WARNING]
> Do not attach patient-identifiable images, real clinical records, generated reports containing private information, API keys, passwords, tokens, or restricted datasets. Use synthetic or fully de-identified examples only.

## Summary

Describe the problem in one or two clear sentences.

## Affected workflow

Select all that apply:

- [ ] Professional app (`app_qadamcare_pro.py`)
- [ ] Visible ulcer-like RGB segmentation
- [ ] STANDUP paired RGB + grayscale thermal analysis
- [ ] Pseudo-colour thermal research analysis
- [ ] Image-quality or input-validation gate
- [ ] Thermal monitoring-zone mapping
- [ ] LLM report generation
- [ ] VLM documentation
- [ ] PDF export
- [ ] Training or evaluation script
- [ ] Documentation or repository setup
- [ ] Other

## Steps to reproduce

1. 
2. 
3. 
4. 

## Expected behaviour

What should have happened?

## Actual behaviour

What happened instead?

## Error output

Paste the smallest useful error message or traceback inside a code block. Remove usernames, local institution paths, patient data, secrets, and tokens.

```text
Paste error here
```

## Input contract

- Selected workflow:
- Input type used: close-up RGB / plantar RGB / grayscale thermal / pseudo-colour thermal / other
- Were RGB and thermal files a confirmed matched pair? yes / no / not applicable
- Did the app mark the input as valid, blocked, or review needed?
- Model/checkpoint involved, if known:

## Environment

- Operating system and version:
- Python version:
- Streamlit version:
- PyTorch and torchvision versions:
- CPU or GPU:
- GPU model and driver, if relevant:
- Ollama version, if relevant:
- Text model, if relevant:
- Vision model, if relevant:
- Git branch and commit:

## Reproducibility

- [ ] Happens every time
- [ ] Happens sometimes
- [ ] Happened once
- [ ] I can reproduce it with a non-sensitive example

## Safety, privacy, or correctness impact

Select all that apply:

- [ ] Incorrect model routing
- [ ] Incorrect or contradictory result
- [ ] Healthy/control-like output described as confirmed diabetes
- [ ] Diabetes history confused with image-model output
- [ ] Unsupported diagnosis or treatment wording
- [ ] Invalid segmentation or thermal visualisation shown as valid
- [ ] LLM/VLM hallucination or contradiction
- [ ] PDF/report mismatch
- [ ] Patient-data or privacy risk
- [ ] Secret or credential exposure
- [ ] No known safety impact

Explain the impact and whether the affected output was shown, downloaded, or shared.

## Logs or screenshots

Attach only de-identified screenshots. Crop out names, IDs, file paths, email addresses, and medical information not needed to reproduce the issue.

## Additional context

Include any attempted fix, related issue, recent change, or useful technical detail.
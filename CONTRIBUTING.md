# Contributing to QadamCare AI

Thank you for contributing to **QadamCare AI**.

QadamCare AI is an educational engineering and research prototype for workflow-safe foot-image analysis, monitoring support, and clinician-facing documentation. Contributions must preserve privacy, reproducibility, transparent limitations, and non-diagnostic wording.

## Before contributing

Read:

- [README.md](README.md)
- [MODEL_CARD.md](MODEL_CARD.md)
- [SECURITY.md](SECURITY.md)
- [DATA_AND_PRIVACY.md](DATA_AND_PRIVACY.md)
- [DISCLAIMER.md](DISCLAIMER.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## Non-negotiable rules

Do not:

- upload patient-identifiable data, private clinical records, or unauthorised images
- commit datasets, checkpoints, reports, `.env`, secrets, tokens, or passwords
- send an image to an incompatible model workflow
- describe the system as diagnosing diabetes or diabetic-foot complications
- describe healthy/control-like as proof that a person is healthy
- describe diabetic-foot-like as proof that a person has diabetes
- present relative thermal intensity as calibrated temperature
- present monitoring zones as future-ulcer predictions
- present R0/R1/R2 as reliable without materially improved validation
- remove warnings, blocked-state behaviour, provenance, or generated-AI disclaimers
- add treatment, medication, surgery, antibiotic, or emergency-triage recommendations

## Architecture rules

Keep these workflows separate:

1. close-up RGB ulcer-like segmentation
2. paired STANDUP plantar RGB + grayscale thermal fusion
3. pseudo-colour thermal-only research analysis

Any new workflow must define:

- accepted input type
- validation gate
- model/checkpoint dependency
- output meaning
- blocked/error behaviour
- safety limitation
- test case using non-sensitive data

## Generative AI rules

LLM/VLM output must:

- use structured application data as the factual source
- distinguish user-entered history from image-model outputs
- preserve supplied numbers
- avoid inventing missing findings
- treat “No” as explicitly absent, not missing
- be labelled as generated and potentially incorrect
- avoid diagnosis, prognosis, severity grading, treatment, and prescribing

Prompt changes should be tested against at least:

- no known diabetes + healthy/control-like pattern
- known diabetes + healthy/control-like pattern
- user-entered red flags with a non-diagnostic model result
- invalid image blocked before inference
- missing checkpoint or Ollama unavailable

## Development workflow

1. Create a focused branch.
2. Make the smallest coherent change.
3. Add or update tests and documentation.
4. Run syntax checks and the affected application workflow.
5. Confirm no sensitive files were created or staged.
6. Review wording for unsupported medical certainty.
7. Open a pull request with evidence of testing.

Suggested branches:

```text
feature/pdf-report-improvements
feature/workflow-validation
fix/diabetes-history-separation
fix/thermal-routing
security/local-file-handling
docs/model-card-update
```

## Pull-request checklist

Include in the PR description:

- purpose and affected workflow
- files changed
- commands used for testing
- screenshots or logs using non-sensitive examples
- model/checkpoint assumptions
- privacy impact
- clinical-safety impact
- documentation updated
- known limitations

Checklist:

```text
[ ] No patient data, secrets, datasets, reports, or checkpoints committed
[ ] Correct workflow routing preserved
[ ] Invalid inputs are blocked safely
[ ] User-entered history is separated from model output
[ ] No diagnostic or treatment claim added
[ ] LLM/VLM output is labelled as generated
[ ] README/model card/security docs updated where needed
[ ] Tests or reproducible manual checks completed
```

## Code quality

- Prefer clear names and small functions.
- Add type hints where practical.
- Handle missing files and incompatible checkpoints without crashing the full app.
- Avoid absolute local paths.
- Keep generated outputs under ignored folders.
- Do not silently fall back to a different model.
- Record thresholds and assumptions near the code that uses them.

## Model and dataset contributions

Before adding a model or dataset reference, document:

- source and citation
- licence or access terms
- intended task
- participant/sample counts where known
- split strategy and leakage prevention
- evaluation metrics
- limitations and known biases
- redistribution rights

Do not commit third-party data or weights simply because they are publicly downloadable.

## Reporting security issues

Use the private process in [SECURITY.md](SECURITY.md). Do not open a public issue containing exploit details, secrets, or sensitive data.
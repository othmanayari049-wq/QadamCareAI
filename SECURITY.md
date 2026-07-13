# Security Policy

## Security Overview

QadamCare AI is an educational engineering prototype related to medical imaging, clinician-facing documentation, and AI-assisted screening-support workflows.

Although this repository is not a production medical system, it must be handled carefully because medical-image projects can involve privacy, security, licensing, and clinical-safety risks.

---

## Supported Scope

This repository is intended for:

- academic use
- engineering demonstration
- supervised research
- prototype development
- non-diagnostic workflow exploration

This repository is not approved for:

- real clinical deployment
- direct patient use
- autonomous diagnosis
- treatment decision-making
- emergency triage
- unsupervised medical screening

---

## Sensitive Data Policy

The following must not be uploaded to this repository:

- patient-identifiable information
- raw patient medical images
- private clinical records
- hospital documents containing sensitive information
- generated reports containing private information
- API keys, tokens, or passwords
- local environment secrets
- restricted model checkpoints
- raw restricted datasets
- temporary files containing private data

This repository should contain code, safe documentation, configuration files without secrets, and approved non-sensitive examples only.

---

## Safe Repository Content

Acceptable repository content includes:

- source code
- non-sensitive documentation
- architecture documentation
- model cards
- safety documentation
- approved demo examples
- configuration files without secrets
- issue templates
- contribution guidelines

The repository should not contain patient data, private medical files, or restricted datasets.

---

## Reporting a Security Issue

If you find a security, privacy, or sensitive-data issue, please report it privately to the repository owner instead of opening a public issue.

Please include:

- a short description of the issue
- the file, page, or module where the issue appears
- why it may affect privacy, safety, or security
- recommended mitigation if known
- whether sensitive data may already have been exposed

---

## Examples of Security or Privacy Issues

Examples include:

- accidentally committed private data
- API keys exposed in code
- private file paths included in generated reports
- generated reports containing personal information
- uploaded images that should not be public
- model checkpoints containing restricted data
- unsafe local file-handling behavior
- unsafe local AI prompt or output behavior

---

## Clinical Safety Concerns

QadamCare AI must not be described as a clinical diagnostic tool.

Please report wording or behavior that suggests:

- diagnosis
- treatment recommendation
- confirmed infection
- confirmed ischemia
- confirmed osteomyelitis
- confirmed Wagner grade
- confirmed ulcer depth
- amputation-risk prediction
- unsupported clinical certainty

Clinical-safety issues should be corrected using cautious wording such as:

```text
screening-support only
documentation-support output
visible wound-like region
ulcer-like region
requires clinician review
not clinically validated
not a diagnostic system

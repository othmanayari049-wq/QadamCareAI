# Contributing to QadamCare AI

Thank you for your interest in contributing to **QadamCare AI**.

QadamCare AI is an educational engineering prototype for diabetic-foot visual screening support, follow-up monitoring, clinician-facing documentation, and research-oriented multimodal AI exploration.

Because this project is related to medical imaging and clinical documentation workflows, all contributions must follow strong privacy, safety, and non-diagnostic wording standards.

---

## Contribution Principles

All contributions should support the following goals:

- maintain technical quality
- improve usability and readability
- preserve clinical safety wording
- avoid unsupported medical claims
- protect privacy and sensitive data
- keep the system modular and understandable
- improve documentation and reproducibility

---

## Important Safety Rules

Contributors must not:

- upload patient-identifiable data
- upload private medical images
- upload raw restricted datasets without permission
- upload generated clinical reports containing sensitive data
- upload API keys, tokens, or secrets
- claim that the system performs diagnosis
- claim that the system confirms infection, ischemia, osteomyelitis, Wagner grade, ulcer depth, or treatment decisions
- remove clinical safety disclaimers from the app, documentation, model card, or reports

QadamCare AI must always be described as a **screening-support and documentation-support prototype**, not as a clinical diagnostic system.

---

## What Can Be Contributed

Acceptable contributions include:

- code improvements
- bug fixes
- user-interface improvements
- report design improvements
- documentation improvements
- model evaluation improvements
- refactoring and modularization
- demo workflow improvements
- explainability features
- safer clinical wording
- additional non-diagnostic research-support modules

---

## Recommended Workflow

Before contributing:

1. Create a new branch or fork the repository.
2. Make focused changes.
3. Test the affected feature.
4. Review the code for readability and safety.
5. Confirm that no sensitive data is included.
6. Commit with a clear message.
7. Open a pull request or share the branch for review.

---

## Branch Naming Suggestions

Use clear branch names such as:

```text
feature/pdf-report-improvements
feature/dashboard-ui-cleanup
fix/thermal-validation-bug
fix/report-generation-error
docs/readme-update
docs/model-card-update
refactor/report-engine
refactor/clinical-support-layer

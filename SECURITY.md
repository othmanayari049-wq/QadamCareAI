# Security Policy

## Project scope

QadamCare AI is an educational engineering and research prototype involving medical-image workflows, local AI models, and generated clinician-facing documentation. It is not approved for clinical deployment, direct patient use, emergency triage, autonomous diagnosis, or treatment decision-making.

Security reports may concern software security, privacy, data exposure, model integrity, prompt injection, unsafe medical wording, or misuse of generated reports.

## Supported versions

Security and safety fixes are maintained on:

- the current development branch used for the professional application
- the latest merged default-branch version

Older experimental scripts may remain in the repository for reproducibility but should not be treated as supported deployment entry points.

## Report a vulnerability privately

Do not open a public issue when the report contains an exploitable detail, secret, patient information, private image, restricted dataset path, or unsafe generated report.

Contact the repository owner privately:

- Email: `othmanayari049@gmail.com`
- GitHub: `@othmanayari049-wq`

Include:

- affected file, branch, commit, or application workflow
- reproducible steps with non-sensitive test data
- expected and observed behaviour
- potential privacy, security, or clinical-safety impact
- suggested mitigation, when known
- whether a secret or sensitive file may already have been exposed

Do not attach real patient data. Use synthetic or fully authorised examples.

## Sensitive-data policy

Never commit or share through this repository:

- patient names, identifiers, dates of birth, medical-record numbers, or contact details
- raw clinical images or reports without explicit approval and an appropriate governance process
- hospital documents, consent forms, or clinical notes
- generated PDFs containing identifiable information
- raw restricted datasets or redistributed copies that violate source terms
- model checkpoints whose redistribution rights are unclear
- `.env`, `.streamlit/secrets.toml`, API keys, passwords, tokens, certificates, or private endpoints
- absolute local paths that reveal usernames, institutions, or private folder structures
- Ollama prompts or logs containing sensitive case details

If sensitive material is committed, stop sharing the repository, revoke exposed credentials, remove the material from Git history, and document the incident privately.

## Local-storage behaviour

QadamCare stores uploaded images and generated reports in local output folders during use. Users are responsible for:

- using non-sensitive or properly authorised images
- restricting access to the local computer
- removing generated files after testing
- avoiding cloud-synchronised folders for sensitive material unless approved
- applying institutional retention and deletion requirements

The application does not provide a hospital-grade data-retention, encryption, authentication, audit-log, backup, or access-control system.

## Local LLM/VLM safety

Ollama runs locally, but local execution does not automatically make a workflow secure or clinically reliable.

Risks include:

- prompt injection from text or images
- hallucinated clinical statements
- stale patient context reused across Streamlit sessions
- local logs or generated files containing sensitive information
- denial of service from oversized prompts or images
- unsafe model outputs presented as verified facts

Generated LLM/VLM text must remain clearly separated from deterministic inputs and model outputs. It must not be used as a diagnosis, prescription, or autonomous clinical decision.

## Model and checkpoint integrity

Only load checkpoints obtained from a trusted source. Before distribution or deployment:

- verify the expected architecture and checkpoint path
- record a cryptographic checksum
- avoid untrusted pickle files
- prefer state-dict formats over arbitrary executable serialisation
- document the training dataset and licence
- test for input-routing errors and out-of-distribution failures

PyTorch checkpoint loading can execute unsafe code when untrusted serialised objects are used. Never load an unknown checkpoint merely because it has a `.pth` or `.pt` extension.

## Application security limitations

The Streamlit app is designed for local demonstration. It currently does not provide production-grade:

- user authentication or role-based access control
- encrypted database storage
- hospital identity integration
- consent management
- immutable audit logging
- secure multi-tenant isolation
- rate limiting or abuse controls
- regulatory compliance guarantees
- monitored production deployment

Do not expose the local app directly to the public internet.

## Clinical-safety issues

Please report any behaviour or wording that:

- treats a healthy/control-like image pattern as proof that the person is healthy
- treats a diabetic-foot-like image pattern as a diabetes diagnosis
- confuses user-entered diabetes history with an image-model result
- sends an image to the wrong workflow or model
- interprets pseudo-colour intensity as calibrated temperature
- presents a monitoring zone as a future-ulcer prediction
- presents R0/R1/R2 as reliable despite weak validation
- confirms infection, ischemia, osteomyelitis, wound depth, severity, Wagner grade, prognosis, amputation risk, or treatment need
- hides model failure, blocked execution, missing checkpoints, or invalid image quality

Preferred wording includes:

```text
screening-support only
dataset-defined image pattern
user-entered history
relative image intensity
visible wound-like or ulcer-like region
review flag
requires clinician assessment
not clinically validated
not a diagnostic system
```

## Dependency and environment security

- Use a dedicated virtual environment.
- Keep Python, PyTorch, Streamlit, Pillow, OpenCV, ReportLab, Requests, and other dependencies updated.
- Review dependency changes before installation.
- Run the app with the minimum local permissions required.
- Do not run Ollama or Streamlit as an administrator unless necessary for a controlled setup task.
- Keep the repository private while it contains unreviewed research code or sensitive configuration.

## Disclosure expectations

The maintainer will try to acknowledge a valid private report promptly, assess impact, prepare a fix, and coordinate disclosure. Response time is not guaranteed because this is a student research project, not a staffed production service.

## No security or clinical warranty

The presence of this policy does not imply that the project is secure, compliant, clinically validated, or suitable for patient care. See [DISCLAIMER.md](DISCLAIMER.md) and [DATA_AND_PRIVACY.md](DATA_AND_PRIVACY.md).
# Data and Privacy Policy

## Purpose

This document defines the minimum data-handling expectations for QadamCare AI. It does not replace institutional privacy, ethics, legal, or cybersecurity requirements.

## Repository policy

The Git repository must not contain:

- patient-identifiable information
- real clinical images without explicit approval
- generated reports containing names, IDs, dates, or medical details
- hospital records or consent documents
- raw datasets that cannot legally be redistributed
- trained checkpoints with unclear redistribution rights
- secrets, credentials, tokens, private keys, or `.env` files
- local absolute paths revealing usernames or institutions

The following folders and file types are ignored by default:

- `data/`
- `outputs/`
- model checkpoint formats
- common medical image formats
- generated PDF and document reports
- environment and secret files

Ignoring a file does not make its local storage secure.

## Local application behaviour

During use, the Streamlit application may write uploaded images, overlays, masks, temporary files, and generated reports to local output folders.

Users must:

- use synthetic, public, de-identified, or properly authorised data
- understand whether their project folder is synchronised to OneDrive, Dropbox, Google Drive, or another cloud service
- restrict filesystem access
- remove generated files after use
- avoid sharing screenshots or reports containing case information
- follow applicable institutional retention requirements

QadamCare AI does not currently provide automatic deletion, encryption at rest, audit logging, authentication, or role-based access control.

## De-identification

Removing a name alone is not sufficient de-identification. Images and reports may still contain:

- faces or body features
- dates and timestamps
- hospital labels
- embedded metadata
- file names containing identifiers
- medical history combinations that allow re-identification

Before authorised research use, apply an institutionally approved de-identification process and verify the output manually.

## Image metadata

JPEG, PNG, TIFF, and thermal-camera files can contain metadata. Before sharing an approved example:

- inspect and remove EXIF or device metadata
- remove identifying labels within the image
- use neutral filenames
- confirm that the background does not reveal identity or location

## LLM/VLM privacy

The configured Ollama models run locally, but prompts and images may still be stored in:

- application output files
- local logs
- terminal history
- screenshots
- operating-system caches
- cloud-synchronised folders

Do not include identifiable patient information in prompts unless an approved governance process specifically permits it.

## Dataset governance

For every dataset, record:

- official source
- version and access date
- licence or terms of use
- required citation
- consent and ethics conditions, where stated
- whether redistribution is permitted
- permitted research purpose
- local storage location and access controls

Do not copy datasets into the repository. Do not publish a combined dataset unless every source explicitly permits redistribution and combination.

## Model checkpoints

Checkpoints can preserve information about training data and may be subject to the dataset’s terms. Before distributing a checkpoint:

- confirm that derivative model distribution is permitted
- document training sources
- remove embedded local paths or metadata
- publish a checksum
- document architecture and expected preprocessing
- test that the checkpoint contains only the expected state dictionary

## Generated reports

Generated PDFs and Markdown reports may contain:

- patient-entered details
- images and overlays
- model probabilities
- clinical findings
- LLM/VLM narratives

Treat generated reports as sensitive whenever they relate to a real person. Store and delete them under the same rules as the source data.

## Incident response

When sensitive information is accidentally committed or shared:

1. stop further sharing
2. notify the project owner and appropriate institutional contact
3. revoke exposed credentials
4. remove the material from the current branch and Git history
5. assess cloud mirrors, forks, and local synchronisation
6. document what was exposed and for how long
7. follow institutional incident-response requirements

See [SECURITY.md](SECURITY.md) for private reporting.

## Clinical and regulatory limitation

This repository does not claim compliance with HIPAA, GDPR, Qatar health-data requirements, medical-device regulations, or hospital cybersecurity standards. Formal legal and institutional review would be required before processing real patient data or deploying the software.
# Third-Party Notices

QadamCare AI includes or interoperates with third-party software, datasets, publications, and local models. Their rights are **not** granted by this repository’s MIT License.

## General rule

Before downloading, training, redistributing, publishing, or deploying any third-party asset, review its official licence and terms. Public availability does not automatically permit redistribution or clinical use.

## Research datasets

### FUSeg / Foot Ulcer Segmentation Challenge

Used for development of the visible ulcer-like RGB segmentation module.

- purpose in this project: binary segmentation research
- redistributed here: no
- checkpoint redistribution: must be reviewed against the dataset terms
- required action: cite the official dataset/challenge and associated publication in academic work

### STANDUP Database

Used for paired plantar RGB and grayscale thermal research.

- project: STANDUP Database
- associated publication: Bouallal et al., *STANDUP Database: a database of paired thermal and visible spectrum plantar-foot images for diabetic-foot research*, Open Research Europe, 2022
- DOI: `10.12688/openreseurope.14706.1`
- purpose in this project: paired image-pattern classification and experimental R0/R1/R2 research
- redistributed here: no
- required action: obtain from an authorised source and follow its citation and use terms

### Other thermal research data

The legacy pseudo-colour thermal workflow may depend on a separately obtained plantar-thermogram dataset. Its images, temperature files, and derived checkpoints are not covered by this repository licence. Document the exact source and permissions before distribution.

## Local generative models

The application can use Ollama with models such as:

- `qwen2.5:3b`
- `qwen2.5vl:3b`

Model licences, acceptable-use terms, and redistribution rules are controlled by their respective publishers and Ollama manifests. The repository does not redistribute those models.

## Python libraries

The project depends on open-source packages including PyTorch, torchvision, OpenCV, Albumentations, segmentation-models-pytorch, NumPy/Pandas-related tooling, scikit-learn, Streamlit, Pillow, Requests, PyYAML, Matplotlib, tqdm, and ReportLab.

Each library remains governed by its own licence. Consult installed package metadata and official repositories for exact terms.

## Publications and clinical guidelines

References to publications or guidelines are provided for research context and documentation. Their text, figures, tables, and recommendations are not relicensed by this repository.

## Trademarks and institutions

Names such as Qatar University, Ollama, PyTorch, Streamlit, and dataset or challenge names may be trademarks or institutional identifiers. Their mention does not imply endorsement, partnership, clinical approval, or sponsorship.

## Contributor responsibility

Contributors adding a dependency, model, dataset, figure, icon, or copied code must:

- identify the source
- verify compatibility with repository use
- preserve attribution and licence notices
- avoid committing restricted assets
- update this file and the model card when relevant

When licence status is unclear, do not redistribute the asset.
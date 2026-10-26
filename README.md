# Continuous property-weighted adaptation for aqueous redox molecular generation

Code and lightweight numerical snapshots for manuscript v66:
**Continuous property-weighted adaptation with retained domain support for aqueous redox molecular generation**.
Authors: Sohyeon Lee, Daewon Chung and Joonhyeon Jeon. Snapshot: `v66-licensed-r1`.

## Code and data locations

This repository contains the68 existing Python source files used in the v66 archive,
selected configuration/provenance files, environment specifications and seven small
current-comparison CSV summaries. Model weights, working molecular corpora, scored
attempts and Gaussian logs are in the four full-data archives listed in
`DATA_ARCHIVES.json`. No new experiment or external result is included.

Zenodo record URL/version DOI: pending actual deposition. No DOI or repository URL
is asserted by this prepared tree. The original ChEMBL release was not recorded;
the full archive supplies exact input hashes and verified preprocessing provenance.

## Existing comparison

The current study has16 paired fine-tuning seeds per backbone,96 runs,864 decoding
cells and864000 saved attempts. Hard10 and Continuous10 share a fixed pretrained
checkpoint within backbone, omit ChEMBL replay and use final epoch10 after8000 updates.
The prespecified primary inference supports the TCN-LSTM contrast; LSTM and TCN
remain unresolved. Other endpoints are descriptive. Earlier controls, sensitivity,
diagnostics and DFT evidence retain their separate protocols in the full archive.

## Reproduce existing figures from included summaries

```bash
python -m pip install -r environment/requirements-analysis.txt
python numerical/current_comparison/render_current_figures.py --output-dir reproduced_figures
```

This plotting command uses saved summaries only. It does not train, generate,
predict molecular properties, run DFT or introduce a statistical family.
The included summaries are the unchanged v66 records. The68 original source files
are retained byte-for-byte; `.gitattributes` disables newline normalization to
preserve recorded hashes. Historical paths/defaults in frozen source are provenance.

## Full reproduction materials

Obtain all four archives named in `DATA_ARCHIVES.json` from the issued Zenodo record
once available. Verify them with:

```bash
python tools/verify_downloads.py --archives-dir /path/to/downloaded/archives
```

Extract all four into a separate directory, such as
`reproduction_workspace/AORFB_v66_public_release/`. Follow that archive's README
and `docs/REPRODUCIBILITY.md` for saved-score checks and the separately documented
model implementation. This GitHub tree is a code/light-summary distribution, not
the complete training or quantum-chemical dataset. Do not extract the full-data
README over this repository's README. Training, generation and DFT replay were not
performed during registration preparation.

## Licenses

Author-owned source code: MIT (`LICENSE`). Author documentation and independent
numerical snapshots: CC BY4.0. `LICENSE_SCOPE.csv` maps this repository's files;
`LICENSES/` contains the legal texts. License texts themselves retain their terms.
The full-data archive separately maps ChEMBL samples/generator weights/generated
structures under CC BY-SA3.0 and original RedDB content under CC0. Those data/model
terms are not replaced by the code license. No new rights in chemical facts are claimed.

## Citation and status

Use `CITATION.cff` and the exact Zenodo version DOI after it is issued. Author names,
manuscript version and the archive hashes are included without invented ORCIDs or
publication dates. Repository creation, external upload and DOI registration have
not been performed by this local preparation.

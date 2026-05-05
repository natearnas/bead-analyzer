# Contributing to Bead Analyzer

Thank you for your interest in contributing! We welcome input from the microscopy, biophysics, and neuroinformatics communities to help make this tool more robust and useful for everyone.

## The Contribution Process

To maintain the scientific integrity and performance of the codebase, all contributions are welcomed through vetted Pull Requests (PRs).

1. **Fork the Repository:** Create your own fork of the project to work on.
2. **Create a Feature Branch:** Keep your changes organized in a dedicated branch (e.g., `feature/improved-deconvolution` or `fix/metadata-parsing`).
3. **Run Tests:** Before submitting, make sure all tests pass:
   ```bash
   pip install -e ".[dev]"
   pytest tests/ -v
   ```
   CI will also run these automatically on your PR across Python 3.9–3.11 (Ubuntu + Windows).
4. **Submit a Pull Request:** Once your changes are tested, submit a PR to the main branch.
5. **The Vetting Process:** Each PR will be reviewed for:
   - **Scientific Accuracy:** Does the math/logic hold up for imaging data?
   - **Code Quality:** Is the code readable and follows standard Python (PEP 8) conventions?
   - **Documentation:** Are new features or changes clearly explained?

## Licensing and Intellectual Property

Arnas Technologies, LLC maintains this project as an open-source tool for the scientific community. To ensure a clear and ethical distinction between community efforts and commercial services:

- **Open Source Grant:** By submitting a Pull Request, you agree to license your contribution under the project's MIT license.
- **Non-Commercial Incorporation:** Arnas Technologies, LLC explicitly commits that community-contributed code will not be included in any separate, closed-source commercial licenses sold by the company.
- **Your Rights:** You retain the copyright to your contributions, which will always remain available to the public under this project's open-source license.

## Cutting a Release (Maintainers)

Each tagged GitHub release is auto-archived to Zenodo via the `.zenodo.json` webhook and assigned a new version-specific DOI (v1.2.1 received `10.5281/zenodo.20031695`). The **concept DOI** (`10.5281/zenodo.20031694`) is stable across versions and always resolves to the latest release, so the README badge and any external citations don't need updating.

Steps:

1. Bump `version` in [pyproject.toml](pyproject.toml) and [CITATION.cff](CITATION.cff); update `date-released` in CITATION.cff to today.
2. Add a new entry at the top of [CHANGELOG.md](CHANGELOG.md) following the Keep a Changelog format (`## [X.Y.Z] - YYYY-MM-DD` with `### Added` / `### Changed` / `### Fixed` subsections as needed).
3. If author list, affiliation, or ORCIDs changed, update [.zenodo.json](.zenodo.json) `creators` and the matching CITATION.cff `authors` block.
4. Commit (e.g. `release X.Y.Z`) and push to `main`.
5. Tag and push the tag:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
6. On GitHub, **Releases → Draft a new release** → choose tag `vX.Y.Z` → title `vX.Y.Z — short summary` → paste only that version's CHANGELOG entry into the description (do not include the file header or older entries) → leave "Set as the latest release" checked, "Set as a pre-release" unchecked → **Publish release**.
7. Within ~1 minute, Zenodo will deposit the source tarball and assign a new version DOI. Verify at https://zenodo.org/account/settings/github/. No further action is needed — the concept DOI in the README and CITATION.cff already points to the new version automatically.

## Support and Questions

If you have a specialized use case or are interested in a deeper collaboration (such as grant inclusion or custom pipeline development), please refer to the [Consulting & Collaboration](README.md#consulting--collaboration) section in the README or contact Nate directly at nate@arnastechnologies.com.

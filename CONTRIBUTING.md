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

## Support and Questions

If you have a specialized use case or are interested in a deeper collaboration (such as grant inclusion or custom pipeline development), please refer to the [Consulting & Collaboration](README.md#consulting--collaboration) section in the README or contact Nate directly at nate@arnastechnologies.com.

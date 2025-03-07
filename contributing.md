# Contribution Guidelines for PyFibers

Thank you for your interest in contributing to **PyFibers**, our open-source Python package for simulating peripheral nerve fiber responses to electrical stimulation using NEURON. Contributions are instrumental in maintaining and advancing this software, facilitating reproducible research, and fostering a vibrant community of users and developers.

Please follow these guidelines to ensure smooth collaboration and maintain high standards of scientific rigor and code quality.

## Table of Contents
- [Getting Started](#getting-started)
- [Contribution Workflow](#contribution-workflow)
- [Code Style and Formatting](#code-guidelines)
- [Documentation](#documentation-guidelines)
- [Testing and Validation](#testing-and-validation)
- [Plugin Development](#creating-and-sharing-fiber-models-as-plugins)
- [Issue Reporting](#issue-reporting)
- [Discussions](#discussions)
- [License and Attribution](#license-and-attribution)

## Getting Started
### Installation for Development

First, fork the repository on GitHub and then clone your fork locally:

```bash
git clone https://github.com/your-username/pyfibers.git
cd pyfibers
```

Install PyFibers locally along with all development dependencies:

```bash
pip install .[dev]
```

Compile NEURON mechanisms:
```bash
pyfibers_compile
```

IMPORTANT: Install `pre-commit` and set up the pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```
Installing `pre-commit` hooks will save you from a headache later on, code that is not compliant with our pre-commit checks *will not be merged*.

## Contribution Workflow

We follow a Git-based workflow:

1. **Fork** the repository.
2. Create a new branch from the latest `main`:

```bash
git checkout -b feat/your-new-feature
```

3. Implement your changes following the [Code Guidelines](#code-guidelines).
4. Write or update tests to cover your changes (see [Testing](#testing-and-validation)).
5. Update documentation as necessary (see [Documentation Guidelines](#documentation-guidelines)).
6. Commit changes using clear messages following the [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/) specification:

    ```
    feat: add new fiber model based on Sweeney 2023
    ```

    - For work-in-progress commits, use `chore`.
    - Branches containing *any* commits that don't follow the Conventional Commit specification will be automatically rejected; installing `pre-commit` hooks will enforce correct formatting. (If necessary, you can use an interactive rebase to fix commit messages.)
    - Commitizen can be used as an interactive CLI tool to help you write conventional commits. Install it using `pip install commitizen` and then use `cz commit` to commit your changes.

7. Push the branch to your fork and create a Pull Request (PR).
8. Collaborate with maintainers by addressing code reviews.

## Code Guidelines

- Make sure your code is well commented and documented.
- Use Sphinx-compatible docstrings (reStructuredText format).
- Code must pass automated formatting and linting checks from `.pre-commit-config.yaml`.

These checks will run automatically if you've installed `pre-commit`. PRs with failing checks will not be merged.

## Testing and Validation

- We use **pytest** and **tox** for testing.
- New features must include comprehensive tests.
- Run tests locally using:

```bash
pytest
```

- All tests must pass before PRs will be merged.
- Continuous Integration will automatically verify your PR on multiple Python and NEURON versions.

## Documentation Guidelines

Documentation consists of:
- **Tutorials and user guides**: Written using [MyST Markdown](https://myst-parser.readthedocs.io/en/latest/) syntax in Markdown files and Jupyter Notebooks.
- **API documentation**: Auto-generated from Sphinx-style docstrings (reStructuredText syntax).

Ensure your contributions update relevant documentation when changing or adding features. Documentation changes must accompany code changes in the same PR.

## Creating and Sharing Fiber Models as Plugins

New fiber models can be published as plugins to PyFibers:

- Follow the [plugin documentation](<<link>>) to create a new fiber model.
- Upload the fiber model plugin as a separate repository.
- The plugin will automatically become available in PyFibers upon installation.

## Issue Reporting

Report bugs, suggest improvements, or request features via:

- **Internal collaborators**: GitLab issues.
- **External users**: GitHub issues.

Provide detailed steps to reproduce the issue, including Python, NEURON, and OS versions, along with error messages and logs.

## Discussions

We encourage active discussion and community engagement:

- Internal discussions: GitLab issues
- External community discussions: GitHub Discussions

Feel free to propose new features, ask questions, and share insights!

## License and Attribution

PyFibers is open-source and distributed under the (PLACEHOLDER) License. Contributions implicitly agree to release their work under this license.

---

**Thank you** for contributing to PyFibers. Your contributions help advance computational modeling in neural engineering and ensure rigorous and reproducible scientific research.

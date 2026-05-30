# Contributing

This is an individual BSc dissertation project. Development is carried out by the author, and code contributions are unlikely to be merged while the work is under assessment, since academic-integrity requirements mean that the submitted work must remain clearly our own. The repository is nonetheless public by design, and several forms of input are valuable:

- **Bug reports.** If something does not run, or runs but produces a clearly incorrect result, we would like to know.
- **Questions about the method.** If part of the federated-learning or distillation setup is unclear, that is usually a sign that the documentation needs improvement.
- **Corrections.** Statistical, methodological, or data-leakage issues are the most valuable feedback of all.

Feature requests are welcome, but the scope is fixed by the dissertation, and most will not be incorporated before submission.

## Reporting an issue

Please open an [issue](https://github.com/vvaitsopoulou/stressfl/issues). A brief check that the problem is not already listed saves duplicated effort. A useful report states what was done, what was expected, what occurred, and the environment in which it occurred (operating system, Python version, and, where possible, the error text). For a numerical or result-related issue, the relevant configuration or the affected fold is particularly helpful.

## Setting up the environment

The setup mirrors the README:

```bash
git clone https://github.com/vvaitsopoulou/stressfl.git
cd stressfl

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

We then recommend installing the pre-commit hooks, since the checks described below are enforced on commit and are far easier to address early:

```bash
pre-commit install
```

## Code style and checks

The pre-commit pipeline runs on every commit and is the authoritative definition of a clean change. It comprises:

- **Black** for formatting (line length 88).
- **Ruff** for linting (replacing flake8, isort, and related tools).
- **MyPy** for type checking.
- **Bandit** for security analysis. This is the reason `torch.load` is called with `weights_only=True`, and why an occasional, justified `# nosec` annotation appears.

The full pipeline can be run manually:

```bash
pre-commit run --all-files
```

or the individual tools:

```bash
black src/
ruff check src/          # add --fix to apply automatic corrections
mypy src/
```

We aim for type hints where they add clarity, Google-style docstrings on non-trivial functions, and small, readable functions in preference to clever ones. Where Ruff reports an unused loop variable, our convention is to rename it with a leading underscore rather than to suppress the rule.

## A note on testing

There is no formal `pytest` suite. As is common for research code, correctness is verified through the exploratory and evaluation notebooks (`notebooks/`) and the verification scripts in `scripts/`, in particular the teacher-checking and leakage-diagnostic scripts. A small, fast, deterministic test is welcome for any new component with a clear correctness criterion, but coverage of the full pipeline is not expected.

## Commit messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification, for example:

```
fix(kd): switch torch.load to weights_only=True to satisfy bandit
feat(fl): add PeFLL hypernetwork client
docs(readme): rewrite structure section to match the actual layout
```

The types in use are `feat`, `fix`, `docs`, `style`, `refactor`, `test`, and `chore`. Subject lines should be concise and written in the imperative mood.

## Pull requests

To contribute a fix, please fork the repository, branch from `develop`, and ensure that `pre-commit run --all-files` passes before opening a pull request. Describe the problem and the change, and reference the relevant issue where one exists. Pull requests will be reviewed, subject to the constraint noted above regarding what can be merged prior to submission.

## Questions

The simplest route is to open an issue, or to contact the author at vaisanna97@gmail.com. Even a brief report — for example, that a command in the README does not work as written — is genuinely helpful.

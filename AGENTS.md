# Repository Guidelines

## Project Structure & Module Organization

This project explores ESS battery cycle-life prediction using the MIT–Stanford battery dataset. The README specifies Batch 1 for training and Batch 2 for evaluation; the EDA examines all three batches.

- `notebooks/ESSHealth-scratch.ipynb`: primary exploratory analysis and plotting workflow.
- `data/README.md`: batch comparisons and analytical findings.
- `data/archive/`: local MATLAB datasets; excluded from Git.
- `images/`: committed analysis plots, typically named `<analysis>_batch<N>.png`.
- `src/` and `results/`: currently empty directories intended for reusable code and model outputs.
- `docs/`: local project documentation; excluded from Git.

## Build, Test, and Development Commands

Use Python 3.11 to match the notebook metadata. Run setup commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` is currently empty. To run the existing notebook, install its dependencies and Jupyter explicitly:

```bash
python -m pip install numpy pandas scipy mat73 matplotlib jupyter
jupyter notebook notebooks/ESSHealth-scratch.ipynb
```

Set the notebook's data-directory path to your local `data/archive/` location before execution. No build pipeline or application entry point is configured.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for Python functions and variables, and descriptive feature names such as `delta_q_var`. Keep notebook cells focused and explain analytical assumptions in Markdown. Move reusable preprocessing and modeling functions into `src/` as the project grows. Prefer configurable paths over machine-specific absolute paths. No formatter or linter is currently configured.

## Testing Guidelines

No automated test suite, testing framework, or coverage threshold exists. Validate notebook changes by restarting the kernel and running all cells sequentially with the required datasets. Check cell counts, missing values, cycle indexing, and regenerated plots. Preserve the documented training/evaluation split and fit preprocessing on training data only. If adding automated tests, place them in `tests/` with `test_*.py` names and document their runner.

## Commit & Pull Request Guidelines

Recent history uses `docs:` messages alongside short Korean descriptions. Prefer concise, scoped messages such as `docs: update batch comparisons` or `feat: add cycle-life features`. Keep changes focused. Pull requests should describe the change, link relevant issues, state datasets and validation performed, and include plot previews when visuals change. Keep raw datasets, virtual environments, and notebook checkpoints out of commits.

# Contributing

Thank you for improving this educational churn-prediction project.

## Development setup

1. Create and activate a Python 3.12-3.14 virtual environment.
2. Install development dependencies with `pip install -r requirements-dev.txt`.
3. Run `python -m pytest` before submitting a change.
4. Run `python src/train.py` and `python src/evaluate.py` when changing the model pipeline.
5. Verify the interface with `streamlit run streamlit_app.py` when changing the app.

## Pull requests

- Keep changes focused and explain their business or technical purpose.
- Add or update tests for preprocessing, prediction logic, and app behavior.
- Do not commit virtual environments, caches, credentials, or customer-sensitive data.
- Update the README and generated reports when model metrics change.

This repository uses the MIT License. By contributing, you agree that your
contribution may be distributed under that license.

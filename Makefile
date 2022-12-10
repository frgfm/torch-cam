# this target runs checks on all files
quality:
	isort . -c
	flake8
	mypy
	pydocstyle
	black --check .
	bandit -r . -c pyproject.toml
	autoflake -r .

# this target runs checks on all files and potentially modifies some of them
style:
	isort .
	black .
	autoflake --in-place -r .

# Run tests for the library
test:
	coverage run -m pytest tests/

# Build documentation for current version
single-docs:
	sphinx-build docs/source docs/_build -a

# Check that docs can build
full-docs:
	cd docs && bash build.sh

# Run the Gradio demo
run-demo:
	streamlit run demo/app.py

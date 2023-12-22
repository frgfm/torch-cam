# this target runs checks on all files
quality:
	ruff format --check .
	ruff check .
	mypy

# this target runs checks on all files and potentially modifies some of them
style:
	ruff format .
	ruff --fix .

# Run tests for the library
test:
	pytest --cov=torchcam tests/ -n auto

# Build documentation for current version
single-docs:
	sphinx-build docs/source docs/_build -a

# Check that docs can build
full-docs:
	cd docs && bash build.sh

# Run the Gradio demo
run-demo:
	streamlit run demo/app.py

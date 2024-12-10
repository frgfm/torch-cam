PYPROJECT_FILE = ./pyproject.toml
DEMO_FILE = ./demo/app.py
TESTS_DIR = ./tests
DOCS_DIR = ./docs

########################################################
# Code checks
########################################################

install-quality: ${PYPROJECT_FILE}
	uv pip install --system -e ".[quality]"
	pre-commit install

lint-check: ${PYPROJECT_FILE}
	ruff format --check . --config ${PYPROJECT_FILE}
	ruff check . --config ${PYPROJECT_FILE}

lint-format: ${PYPROJECT_FILE}
	ruff format . --config ${PYPROJECT_FILE}
	ruff check --fix . --config ${PYPROJECT_FILE}

precommit: ${PYTHON_CONFIG_FILE} .pre-commit-config.yaml
	pre-commit run --all-files

typing-check: ${PYPROJECT_FILE}
	mypy --config-file ${PYPROJECT_FILE}

deps-check: .github/verify_deps_sync.py
	python .github/verify_deps_sync.py

# this target runs checks on all files
quality: lint-check typing-check deps-check

style: lint-format precommit

########################################################
# Build & tests
########################################################

install-test: ${PYPROJECT_FILE}
	uv pip install --system -e ".[test]"

# Run tests for the library
test: ${TESTS_DIR}
	pytest --cov=torchcam ${TESTS_DIR}

install-docs: ${PYPROJECT_FILE}
	uv pip install --system -e ".[docs]"

# Build documentation for current version
single-docs: ${DOCS_DIR}
	sphinx-build ${DOCS_DIR}/source ${DOCS_DIR}/_build -a

# Check that docs can build
full-docs: ${DOCS_DIR}
	cd ${DOCS_DIR} && bash build.sh

install-demo: ${PYPROJECT_FILE}
	uv pip install --system -e ".[demo]"

# Run the Gradio demo
run-demo: ${DEMO_FILE}
	streamlit run ${DEMO_FILE}

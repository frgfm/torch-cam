PY_DIR = .
PACKAGE_DIR = ${PY_DIR}/torchcam
PYPROJECT_FILE = ${PY_DIR}/pyproject.toml
PYTHON_REQ_FILE = /tmp/requirements.txt
DEMO_FILE = ./demo/app.py
TESTS_DIR = ./tests
DOCS_DIR = ./mkdoc



.PHONY: help install install-quality lint-check lint-format precommit typing-check deps-check quality style init-gh-labels init-gh-settings install-mintlify start-mintlify

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

########################################################
# Install & Setup
########################################################

venv:
	uv venv --python 3.11

install: ${PY_DIR} ${PYPROJECT_FILE} ## Install the core library
	uv pip install -e ${PY_DIR}

########################################################
# Code checks
########################################################

install-quality: ${PY_DIR} ${PYPROJECT_FILE} ## Install with quality dependencies
	uv pip install -e '${PY_DIR}[quality]'

lint-check: ${PYPROJECT_FILE} ## Check code formatting and linting
	ruff format --check . --config ${PYPROJECT_FILE}
	ruff check . --config ${PYPROJECT_FILE}

lint-format: ${PYPROJECT_FILE} ## Format code and fix linting issues
	ruff format . --config ${PYPROJECT_FILE}
	ruff check --fix . --config ${PYPROJECT_FILE}

precommit: ${PYPROJECT_FILE} .pre-commit-config.yaml ## Run pre-commit hooks
	pre-commit run --all-files

typing-check: ${PYPROJECT_FILE} ## Check type annotations
	uvx ty check .

deps-check: .github/verify_deps_sync.py ## Check dependency synchronization
	uv run --script .github/verify_deps_sync.py

# this target runs checks on all files
quality: lint-check typing-check deps-check ## Run all quality checks

style: precommit ## Format code and run pre-commit hooks

########################################################
# Builds
########################################################

set-version: ${PYPROJECT_FILE} ## Set the version in the pyproject.toml file
	uv version --frozen --no-build ${BUILD_VERSION}

build: ${PYPROJECT_FILE} ## Build the package
	uv build ${PY_DIR}

publish: ${PY_DIR} ## Publish the package to PyPI
	uv publish --trusted-publishing always

########################################################
# Tests
########################################################

install-test: ${PY_DIR} ${PYPROJECT_FILE} ## Install with test dependencies
	uv pip install -e '${PY_DIR}[test]'

test: ${PYPROJECT_FILE} ## Run the tests
	uv run pytest --cov-report xml


########################################################
# Docs
########################################################

install-docs: ${PYPROJECT_FILE}
	uv pip install -e ".[docs]"

# Build documentation for current version
serve-docs: ${DOCS_DIR}
	DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run mkdocs serve -f ${DOCS_DIR}/mkdocs.yml

# Check that docs can build
build-docs: ${DOCS_DIR}
	DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run mkdocs build -f ${DOCS_DIR}/mkdocs.yml

push-docs: ${DOCS_DIR}
	DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run mkdocs gh-deploy -f ${DOCS_DIR}/mkdocs.yml --force


########################################################
# Demo
########################################################

install-demo: ${PYPROJECT_FILE}
	uv pip install -e ".[demo]"

# Run the Gradio demo
run-demo: ${DEMO_FILE}
	uv run streamlit run ${DEMO_FILE}

########################################################
# Local setup
########################################################

# Push secrets to GH for deployment
push-secrets: .env
	gh secret set -f .env --app actions
	gh secret set -f .env --app dependabot
	gh secret set -f .env --app codespaces

# this target runs checks on all files
quality:
	isort . -c -v
	flake8 ./
	mypy
	pydocstyle torchcam/
	black --check .

# this target runs checks on all files and potentially modifies some of them
style:
	isort .
	black .

# Run tests for the library
test:
	coverage run -m pytest tests/

# Check that docs can build
docs:
	cd docs && bash build.sh

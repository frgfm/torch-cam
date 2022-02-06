# this target runs checks on all files
quality:
	isort . -c -v
	flake8 ./
	mypy torchcam/
	pydocstyle torchcam/

# this target runs checks on all files and potentially modifies some of them
style:
	isort .

# Run tests for the library
test:
	coverage run -m pytest tests/

# Check that docs can build
docs:
	cd docs && bash build.sh

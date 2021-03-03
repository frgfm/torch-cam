# Contributing to torchcam

Everything you need to know to contribute efficiently to the project.



## Codebase structure

- [torchcam](https://github.com/frgfm/torch-cam/blob/master/torchcam) - The actual torchcam library
- [test](https://github.com/frgfm/torch-cam/blob/master/test) - Python unit tests
- [docs](https://github.com/frgfm/torch-cam/blob/master/docs) - Sphinx documentation building
- [scripts](https://github.com/frgfm/torch-cam/blob/master/scripts) - Example and utilities scripts



## Continuous Integration

This project uses the following integrations to ensure proper codebase maintenance:

- [Github Worklow](https://help.github.com/en/actions/configuring-and-managing-workflows/configuring-a-workflow) - run jobs for package build and coverage
- [Codacy](https://www.codacy.com/) - analyzes commits for code quality
- [Codecov](https://codecov.io/) - reports back coverage results

As a contributor, you will only have to ensure coverage of your code by adding appropriate unit testing of your code.



## Issues

Use Github [issues](https://github.com/frgfm/torch-cam/issues) for feature requests, or bug reporting. When doing so, use issue templates whenever possible and provide enough information for other contributors to jump in.



## Developping torchcam


### Commits

- **Code**: ensure to provide docstrings to your Python code. In doing so, please follow [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) so it can ease the process of documentation later.
- **Commit message**: please follow [Udacity guide](http://udacity.github.io/git-styleguide/)

### Running CI verifications locally

#### Unit tests

In order to run the same unit tests as the CI workflows, you can run unittests locally:

```shell
pytest test/
```

#### Lint verification

To ensure that your incoming PR complies with the lint settings, you need to install [flake8](https://flake8.pycqa.org/en/latest/) and run the following command from the repository's root folder:

```shell
flake8 ./
```
This will read the `.flake8` setting file and let you know whether your commits need some adjustments.

#### Annotation typing

Additionally, to catch type-related issues and have a cleaner codebase, annotation typing are expected. After installing [mypy](https://github.com/python/mypy), you can run the verifications as follows:

```shell
mypy --config-file mypy.ini
```
The `mypy.ini` file will be read to check your typing.

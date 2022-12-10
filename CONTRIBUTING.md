# Contributing to torchcam

Everything you need to know to contribute efficiently to the project!

Whatever the way you wish to contribute to the project, please respect the [code of conduct](CODE_OF_CONDUCT.md).



## Codebase structure

- [torchcam](https://github.com/frgfm/torch-cam/blob/main/torchcam) - The actual torchcam library
- [tests](https://github.com/frgfm/torch-cam/blob/main/tests) - Python unit tests
- [docs](https://github.com/frgfm/torch-cam/blob/main/docs) - Sphinx documentation building
- [scripts](https://github.com/frgfm/torch-cam/blob/main/scripts) - Example and utilities scripts
- [demo](https://github.com/frgfm/torch-cam//blob/main/demo) - Small demo app to showcase TorchCAM capabilities



## Continuous Integration

This project uses the following integrations to ensure proper codebase maintenance:

- [Github Worklow](https://help.github.com/en/actions/configuring-and-managing-workflows/configuring-a-workflow) - run jobs for package build and coverage
- [Codacy](https://www.codacy.com/) - analyzes commits for code quality
- [Codecov](https://codecov.io/) - reports back coverage results

As a contributor, you will only have to ensure coverage of your code by adding appropriate unit testing of your code.



## Feedback

### Feature requests & bug report

Whether you encountered a problem, or you have a feature suggestion, your input has value and can be used by contributors to reference it in their developments. For this purpose, we advise you to use Github [issues](https://github.com/frgfm/torch-cam/issues).

First, check whether the topic wasn't already covered in an open / closed issue. If not, feel free to open a new one! When doing so, use issue templates whenever possible and provide enough information for other contributors to jump in.

### Questions

If you are wondering how to do something with TorchCAM, or a more general question, you should consider checking out Github [discussions](https://github.com/frgfm/torch-cam/discussions). See it as a Q&A forum, or the TorchCAM-specific StackOverflow!



## Submitting a Pull Request

### Preparing your local branch

1 - Fork this [repository](https://github.com/frgfm/torch-cam) by clicking on the "Fork" button at the top right of the page. This will create a copy of the project under your GitHub account (cf. [Fork a repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo)).

2 - [Clone your fork](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) to your local disk and set the upstream to this repo
```shell
git clone git@github.com:<YOUR_GITHUB_ACCOUNT>/torch-cam.git
cd torch-cam
git remote add upstream https://github.com/frgfm/torch-cam.git
```

3 - You should not work on the `main` branch, so let's create a new one
```shell
git checkout -b a-short-description
```

4 - You only have to set your development environment now. First uninstall any existing installation of the library with `pip uninstall torchcam`, then:
```shell
pip install -e ".[dev]"
```

### Developing your feature

#### Commits

- **Code**: ensure to provide docstrings to your Python code. In doing so, please follow [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) so it can ease the process of documentation later.
- **Commit message**: please follow [Udacity guide](http://udacity.github.io/git-styleguide/)

#### Unit tests

In order to run the same unit tests as the CI workflows, you can run unittests locally:

```shell
make test
```

#### Code quality

To run all quality checks together

```shell
make quality
```

##### Lint verification

To ensure that your incoming PR complies with the lint settings, you need to install [flake8](https://flake8.pycqa.org/en/latest/) and run the following command from the repository's root folder:

```shell
flake8 ./
```
This will read the `.flake8` setting file and let you know whether your commits need some adjustments.

##### Import order

In order to ensure there is a common import order convention, run [isort](https://github.com/PyCQA/isort) as follows:

```shell
isort **/*.py
```
This will reorder the imports of your local files.

##### Annotation typing

Additionally, to catch type-related issues and have a cleaner codebase, annotation typing are expected. After installing [mypy](https://github.com/python/mypy), you can run the verifications as follows:

```shell
mypy
```
The `pyproject.toml` file will be read to check your typing.

##### Code formatting

Finally, code formatting is a good practice for shareable projects. After installing [black](https://github.com/psf/black), you can run the verifications as follows:

```shell
black .
```

### Submit your modifications

Push your last modifications to your remote branch
```shell
git push -u origin a-short-description
```

Then [open a Pull Request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) from your fork's branch. Follow the instructions of the Pull Request template and then click on "Create a pull request".

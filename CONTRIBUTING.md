# Contributing to torchcam

Everything you need to know to contribute efficiently to the project.



## Codebase structure

- [torchcam](https://github.com/frgfm/torch-cam/blob/master/torchcam) - The actual torchcam library
- [test](https://github.com/frgfm/torch-cam/blob/master/test) - Python unit tests



## Continuous Integration

This project uses the following integrations to ensure proper codebase maintenance:

- [CircleCI](https://circleci.com/) - run jobs for package build and coverage
- [Codacy](https://www.codacy.com/) - analyzes commits for code quality
- [Codecov](https://codecov.io/) - reports back coverage results

As a contributor, you will only have to ensure coverage of your code by adding appropriate unit testing of your code.



## Issues

Use Github [issues](https://github.com/frgfm/torch-cam/issues) for feature requests, or bug reporting. When doing so, use issue templates whenever possible and provide enough information for other contributors to jump in.



## Developping torchcam


### Commits

- **Code**: ensure to provide docstrings to your Python code. In doing so, please follow [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) so it can ease the process of documentation later.
- **Commit message**: please follow [Udacity guide](http://udacity.github.io/git-styleguide/)

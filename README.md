
# Torchcam: class activation explorer

[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/5124b1dff75e4e9cbb68136516605032)](https://www.codacy.com/manual/frgfm/torch-cam?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/torch-cam&amp;utm_campaign=Badge_Grade)  ![Build Status](https://github.com/frgfm/torch-cam/workflows/python-package/badge.svg) [![codecov](https://codecov.io/gh/frgfm/torch-cam/branch/master/graph/badge.svg)](https://codecov.io/gh/frgfm/torch-cam) [![Docs](https://img.shields.io/badge/docs-available-blue.svg)](https://frgfm.github.io/torch-cam)  [![Pypi](https://img.shields.io/badge/pypi-v0.1.1-blue.svg)](https://pypi.org/project/torchcam/) 

Simple way to leverage the class-specific activation of convolutional layers in PyTorch.

![gradcam_sample](static/images/cam_example.png)



## Table of Contents

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Technical Roadmap](#technical-roadmap)
* [Documentation](#documentation)
* [Contributing](#contributing)
* [Credits](#credits)
* [License](#license)



## Getting started

### Prerequisites

- Python 3.6 (or more recent)
- [pip](https://pip.pypa.io/en/stable/)

### Installation

You can install the package using [pypi](https://pypi.org/project/torch-cam/) as follows:

```shell
pip install torchcam
```

or using [conda](https://anaconda.org/frgfm/torchcam):

```shell
conda install -c frgfm torchcam
```



## Usage

You can find a detailed example below to retrieve the CAM of a specific class on a resnet architecture.

```shell
python scripts/cam_example.py --model resnet50 --class-idx 232
```

![gradcam_sample](static/images/cam_example.png)





## Technical roadmap

The project is currently under development, here are the objectives for the next releases:

- [x] Parallel CAMs: enable batch processing.
- [x] Benchmark: compare class activation map computations for different architectures.
- [ ] Signature improvement: retrieve automatically the specific required layer names.
- [ ] Refined RPN: create a region proposal network using CAM.
- [ ] Task transfer: turn a well-trained classifier into an object detector.



## Documentation

The full package documentation is available [here](https://frgfm.github.io/torch-cam/) for detailed specifications. The documentation was built with [Sphinx](sphinx-doc.org) using a [theme](github.com/readthedocs/sphinx_rtd_theme) provided by [Read the Docs](readthedocs.org).



## Contributing

Please refer to `CONTRIBUTING` if you wish to contribute to this project.



## Credits

This project is developed and maintained by the repo owner, but the implementation was based on the following precious papers:

- [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150): the original CAM paper
- [Grad-CAM](https://arxiv.org/abs/1610.02391): GradCAM paper, generalizing CAM to models without global average pooling. 
- [Grad-CAM++](https://arxiv.org/abs/1710.11063): improvement of GradCAM++ for more accurate pixel-level contribution to the activation.
- [Smooth Grad-CAM++](https://arxiv.org/abs/1908.01224): SmoothGrad mechanism coupled with GradCAM.
- [Score-CAM](https://arxiv.org/abs/1910.01279): score-weighting of class activation for better interpretability.
- [SS-CAM](https://arxiv.org/abs/2006.14255): SmoothGrad mechanism coupled with Score-CAM.



## License

Distributed under the MIT License. See `LICENSE` for more information.
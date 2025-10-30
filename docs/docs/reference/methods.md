# Interpretability methods

## Class activation map

The class activation map gives you the importance of each region of a feature map on a model's output.
More specifically, a class activation map is relative to:

* the layer at which it is computed (e.g. the N-th layer of your model)
* the model's classification output (e.g. the raw logits of the model)
* the class index to focus on

With TorchCAM, the target layer is selected when you create your CAM extractor. You will need to pass the model logits to the extractor and a class index for it to do its magic!

## Activation-based methods

Methods related to activation-based class activation maps.

::: torchcam.methods.CAM

::: torchcam.methods.ScoreCAM

::: torchcam.methods.SSCAM

::: torchcam.methods.ISCAM

## Gradient-based methods

Methods related to gradient-based class activation maps.

::: torchcam.methods.GradCAM

::: torchcam.methods.GradCAMpp

::: torchcam.methods.SmoothGradCAMpp

::: torchcam.methods.XGradCAM

::: torchcam.methods.LayerCAM

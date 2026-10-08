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

::: torchcam.methods
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - CAM
            - ScoreCAM
            - SSCAM
            - ISCAM


## Gradient-based methods

Methods related to gradient-based class activation maps.

::: torchcam.methods
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - FinerCAM
            - GradCAM
            - GradCAMpp
            - SmoothGradCAMpp
            - XGradCAM
            - LayerCAM
            - LeGrad
            - RefineCAM

## Token activation maps

Visual explanations for selected vocabulary tokens in multimodal language models. TAM consumes final language-model
states and a linear vocabulary head directly; it does not use the class CAM hook API.

::: torchcam.methods.TAM
    options:
        heading_level: 3
        show_root_heading: true
        members: [__call__]

::: torchcam.methods.DEXAR
    options:
        heading_level: 3
        show_root_heading: true
        members: [__call__, aggregate]

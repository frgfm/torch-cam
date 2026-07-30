# Evaluation metrics

Apart from qualitative visual comparison, it is important to have a refined evaluation metric for class activation maps. This submodule is dedicated to the evaluation of CAM methods.

![Average Drop and Increase in Confidence compare the selected-class score on the original and CAM-masked inputs.](../img/classification-metrics.svg)

::: torchcam.metrics.ClassificationMetric
    options:
        members:
            - reset
            - update
            - summary

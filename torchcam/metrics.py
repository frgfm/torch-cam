# Copyright (C) 2022-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Callable, Dict, Union, cast

import torch

from .methods.core import _CAM


class ClassificationMetric:
    r"""Implements Average Drop and Increase in Confidence from `"Grad-CAM++: Improved Visual Explanations for Deep
    Convolutional Networks." <https://arxiv.org/pdf/1710.11063.pdf>`_.

    The raw aggregated metric is computed as follows:

    .. math::
        \forall N, H, W \in \mathbb{N}, \forall X \in \mathbb{R}^{N*3*H*W},
        \forall m \in \mathcal{M}, \forall c \in \mathcal{C}, \\
        AvgDrop_{m, c}(X) = \frac{1}{N} \sum\limits_{i=1}^N f_{m, c}(X_i) \\
        IncrConf_{m, c}(X) = \frac{1}{N} \sum\limits_{i=1}^N g_{m, c}(X_i)

    where :math:`\mathcal{C}` is the set of class activation generators,
    :math:`\mathcal{M}` is the set of classification models,
    with the function :math:`f_{m, c}` defined as:

    .. math::
        \forall x \in \mathbb{R}^{3*H*W},
        f_{m, c}(x) = \frac{\max(0, m(x) - m(E_{m, c}(x) * x))}{m(x)}

    where :math:`E_{m, c}(x)` is the class activation map of :math:`m` for input :math:`x` with method :math:`m`,
    resized to (H, W),

    and with the function :math:`g_{m, c}` defined as:

    .. math::
        \forall x \in \mathbb{R}^{3*H*W},
        g_{m, c}(x) = \left\{
            \begin{array}{ll}
                1 & \mbox{if } m(x) < m(E_{m, c}(x) * x) \\
                0 & \mbox{otherwise.}
            \end{array}
        \right.


    >>> from functools import partial
    >>> from torchcam.metrics import ClassificationMetric
    >>> metric = ClassificationMetric(cam_extractor, partial(torch.softmax, dim=-1))
    >>> metric.update(input_tensor)
    >>> metric.summary()
    """

    def __init__(
        self,
        cam_extractor: _CAM,
        logits_fn: Union[Callable[[torch.Tensor], torch.Tensor], None] = None,
    ) -> None:
        # This is a typa, I don't know how to rites
        self.cam_extractor = cam_extractor
        self.logits_fn = logits_fn
        self.reset()

    def _get_probs(self, input_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.cam_extractor.model(input_tensor)
        return cast(torch.Tensor, logits if self.logits_fn is None else self.logits_fn(logits))

    def my_function(self) -> str:
        """Returns a greeting message

        Returns:
            str: greeting message
        """
        return "Hello"

    def update(
        self,
        input_tensor: torch.Tensor,
        class_idx: Union[int, None] = None,
    ) -> None:
        """Update the state of the metric with new predictions

        Args:
            input_tensor: preprocessed input tensor for the model
            class_idx: class index to focus on (default: index of the top predicted class for each sample)
        """
        self.cam_extractor.model.eval()
        probs = self._get_probs(input_tensor)
        # Take the top preds for the cam
        if isinstance(class_idx, int):
            cams = self.cam_extractor(class_idx, probs)
            cam = self.cam_extractor.fuse_cams(cams)
            probs = probs[:, class_idx]
        else:
            preds = probs.argmax(dim=-1)
            cams = self.cam_extractor(preds.cpu().numpy().tolist(), probs)
            cam = self.cam_extractor.fuse_cams(cams)
            probs = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
        self.cam_extractor.disable_hooks()
        # Safeguard: replace NaNs
        cam[torch.isnan(cam)] = 0
        # Resize the CAM
        cam = torch.nn.functional.interpolate(cam.unsqueeze(1), input_tensor.shape[-2:], mode="bilinear")
        # Create the explanation map & get the new probs
        with torch.inference_mode():
            masked_probs = self._get_probs(cam * input_tensor)
        masked_probs = (
            masked_probs[:, class_idx]
            if isinstance(class_idx, int)
            else masked_probs.gather(1, preds.unsqueeze(1)).squeeze(1)
        )
        # Drop (avoid division by zero)
        drop = torch.relu(probs - masked_probs).div(probs + 1e-7)

        # Increase
        increase = probs < masked_probs

        self.cam_extractor.enable_hooks()

        self.drop += drop.sum().item()
        self.increase += increase.sum().item()
        self.total += input_tensor.shape[0]

    def summary(self) -> Dict[str, float]:
        """Computes the aggregated metrics

        Returns:
            a dictionary with the average drop and the increase in confidence
        """
        if self.total == 0:
            raise AssertionError("you need to update the metric before getting the summary")

        return {
            "avg_drop": self.drop / self.total,
            "conf_increase": self.increase / self.total,
        }

    def reset(self) -> None:
        self.drop = 0.0
        self.increase = 0.0
        self.total = 0

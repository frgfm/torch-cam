# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import logging
import math
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .core import _CAM
from .utils import locate_linear_layer

__all__ = ['CAM', 'ScoreCAM', 'SSCAM', 'ISCAM']


class CAM(_CAM):
    r"""Implements a class activation map extractor as described in `"Learning Deep Features for Discriminative
    Localization" <https://arxiv.org/pdf/1512.04150.pdf>`_.

    The Class Activation Map (CAM) is defined for image classification models that have global pooling at the end
    of the visual feature extraction block. The localization map is computed as follows:

    .. math::
        L^{(c)}_{CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`,
    and :math:`w_k^{(c)}` is the weight corresponding to class :math:`c` for unit :math:`k` in the fully
    connected layer..

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import CAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = CAM(model, 'layer4', 'fc')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        fc_layer: either the fully connected layer itself or its name
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[nn.Module, str]] = None,
        fc_layer: Optional[Union[nn.Module, str]] = None,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:

        if isinstance(target_layer, list):
            raise TypeError("invalid argument type for `target_layer`")

        super().__init__(model, target_layer, input_shape, **kwargs)

        if isinstance(fc_layer, str):
            fc_name = fc_layer
        # Find the location of the module
        elif isinstance(fc_layer, nn.Module):
            fc_name = self._resolve_layer_name(fc_layer)
        # If the layer is not specified, try automatic resolution
        elif fc_layer is None:
            fc_name = locate_linear_layer(model)  # type: ignore[assignment]
            # Warn the user of the choice
            if isinstance(fc_name, str):
                logging.warning(f"no value was provided for `fc_layer`, thus set to '{fc_name}'.")
            else:
                raise ValueError("unable to resolve `fc_layer` automatically, please specify its value.")
        else:
            raise TypeError("invalid argument type for `fc_layer`")
        # Softmax weight
        self._fc_weights = self.submodule_dict[fc_name].weight.data
        # squeeze to accomodate replacement by Conv1x1
        if self._fc_weights.ndim > 2:
            self._fc_weights = self._fc_weights.view(*self._fc_weights.shape[:2])

    def _get_weights(self, class_idx: int, scores: Optional[Tensor] = None) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps"""

        # Take the FC weights of the target class
        return [self._fc_weights[class_idx, :]]


class ScoreCAM(_CAM):
    r"""Implements a class activation map extractor as described in `"Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks" <https://arxiv.org/pdf/1910.01279.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Score-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)})
        \\odot X

    where :math:`\\odot` refers to the element-wise multiplication and :math:`U` is the upsampling operation.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import ScoreCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ScoreCAM(model, 'layer4')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        batch_size: batch size used to forward masked inputs
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        batch_size: int = 32,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:

        super().__init__(model, target_layer, input_shape, **kwargs)

        # Input hook
        self.hook_handles.append(model.register_forward_pre_hook(self._store_input))
        self.bs = batch_size
        # Ensure ReLU is applied to CAM before normalization
        self._relu = True

    def _store_input(self, module: nn.Module, input: Tensor) -> None:
        """Store model input tensor"""

        if self._hooks_enabled:
            self._input = input[0].data.clone()

    @torch.no_grad()
    def _get_score_weights(self, activations: List[Tensor], class_idx: int) -> List[Tensor]:

        # Initialize weights
        weights = [
            torch.zeros(t.shape[0], dtype=t.dtype).to(device=t.device)
            for t in activations
        ]

        for idx, act in enumerate(activations):
            # Process by chunk (GPU RAM limitation)
            for _idx in range(math.ceil(weights[idx].shape[0] / self.bs)):

                _slice = slice(_idx * self.bs, min((_idx + 1) * self.bs, weights[idx].shape[0]))
                # Get the softmax probabilities of the target class
                weights[idx][_slice] = F.softmax(self.model(act[_slice]), dim=1)[:, class_idx]

        return weights

    def _get_weights(self, class_idx: int, scores: Optional[Tensor] = None) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps"""

        # Normalize the activation
        upsampled_a = [self._normalize(act, act.ndim - 2) for act in self.hook_a]

        # Upsample it to input_size
        # 1 * O * M * N
        spatial_dims = self._input.ndim - 2
        interpolation_mode = 'bilinear' if spatial_dims == 2 else 'trilinear' if spatial_dims == 3 else 'nearest'
        upsampled_a = [
            F.interpolate(up_a, self._input.shape[2:], mode=interpolation_mode, align_corners=False)
            for up_a in upsampled_a
        ]

        # Use it as a mask
        # O * I * H * W
        upsampled_a = [up_a.squeeze(0).unsqueeze(1) * self._input for up_a in upsampled_a]

        # Disable hook updates
        self._hooks_enabled = False
        # Switch to eval
        origin_mode = self.model.training
        self.model.eval()

        weights = self._get_score_weights(upsampled_a, class_idx)

        # Reenable hook updates
        self._hooks_enabled = True
        # Put back the model in the correct mode
        self.model.training = origin_mode

        return weights

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(batch_size={self.bs})"


class SSCAM(ScoreCAM):
    r"""Implements a class activation map extractor as described in `"SS-CAM: Smoothed Score-CAM for
    Sharper Visual Feature Localization" <https://arxiv.org/pdf/2006.14255.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{SS-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\frac{1}{N} \\sum\\limits_1^N softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`N` is the number of samples used to smooth the weights,
    :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\Bigg(\\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)} +
        \\delta\\Bigg) \\odot X

    where :math:`\\odot` refers to the element-wise multiplication, :math:`U` is the upsampling operation,
    :math:`\\delta \\sim \\mathcal{N}(0, \\sigma^2)` is the random noise that follows a 0-mean gaussian distribution
    with a standard deviation of :math:`\\sigma`.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import SSCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = SSCAM(model, 'layer4')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        batch_size: batch size used to forward masked inputs
        num_samples: number of noisy samples used for weight computation
        std: standard deviation of the noise added to the normalized activation
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        batch_size: int = 32,
        num_samples: int = 35,
        std: float = 2.0,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:

        super().__init__(model, target_layer, batch_size, input_shape, **kwargs)

        self.num_samples = num_samples
        self.std = std
        self._distrib = torch.distributions.normal.Normal(0, self.std)

    @torch.no_grad()
    def _get_score_weights(self, activations: List[Tensor], class_idx: int) -> List[Tensor]:

        # Initialize weights
        weights = [
            torch.zeros(t.shape[0], dtype=t.dtype).to(device=t.device)
            for t in activations
        ]

        for idx, act in enumerate(activations):
            # Process by chunk (GPU RAM limitation)
            for _ in range(self.num_samples):
                noisy_m = self._input * (act +
                                         self._distrib.sample(self._input.size()).to(device=self._input.device))

                # Process by chunk (GPU RAM limitation)
                for _idx in range(math.ceil(weights[idx].shape[0] / self.bs)):

                    _slice = slice(_idx * self.bs, min((_idx + 1) * self.bs, weights[idx].shape[0]))
                    # Get the softmax probabilities of the target class
                    weights[idx][_slice] += F.softmax(self.model(noisy_m[_slice]), dim=1)[:, class_idx]

        return [weight.div_(self.num_samples) for weight in weights]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(batch_size={self.bs}, num_samples={self.num_samples}, std={self.std})"


class ISCAM(ScoreCAM):
    r"""Implements a class activation map extractor as described in `"IS-CAM: Integrated Score-CAM for axiomatic-based
    explanations" <https://arxiv.org/pdf/2010.03023.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{ISS-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\sum\\limits_{i=1}^N \\frac{i}{N} softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`N` is the number of samples used to smooth the weights,
    :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\Bigg(\\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)} +
        \\delta\\Bigg) \\odot X

    where :math:`\\odot` refers to the element-wise multiplication, :math:`U` is the upsampling operation,
    :math:`\\delta \\sim \\mathcal{N}(0, \\sigma^2)` is the random noise that follows a 0-mean gaussian distribution
    with a standard deviation of :math:`\\sigma`.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import ISSCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ISCAM(model, 'layer4')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        batch_size: batch size used to forward masked inputs
        num_samples: number of noisy samples used for weight computation
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        batch_size: int = 32,
        num_samples: int = 10,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:

        super().__init__(model, target_layer, batch_size, input_shape, **kwargs)

        self.num_samples = num_samples

    @torch.no_grad()
    def _get_score_weights(self, activations: List[Tensor], class_idx: int) -> List[Tensor]:

        # Initialize weights
        weights = [
            torch.zeros(t.shape[0], dtype=t.dtype).to(device=t.device)
            for t in activations
        ]

        for idx, act in enumerate(activations):
            fmap = torch.zeros((act.shape[0], *self._input.shape[1:]), dtype=act.dtype, device=act.device)
            # Masked input
            mask = act * self._input
            # Process by chunk (GPU RAM limitation)
            for sidx in range(self.num_samples):
                fmap += (sidx + 1) / self.num_samples * self._input * mask

                # Process by chunk (GPU RAM limitation)
                for _idx in range(math.ceil(weights[idx].shape[0] / self.bs)):

                    _slice = slice(_idx * self.bs, min((_idx + 1) * self.bs, weights[idx].shape[0]))
                    # Get the softmax probabilities of the target class
                    weights[idx][_slice] += F.softmax(self.model(fmap[_slice]), dim=1)[:, class_idx]

        return [weight.div_(self.num_samples) for weight in weights]

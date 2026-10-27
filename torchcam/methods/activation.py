# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import logging
import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .core import _CAM, OutputTarget, _resolve_targets, _target_scores

__all__ = ["CAM", "ISCAM", "SSCAM", "EigenCAM", "ScoreCAM"]

logger = logging.getLogger(__name__)


def _svd_right_vectors(matrix: Tensor, *, cpu_fallback: bool) -> Tensor:
    # Some MPS builds have no SVD kernel: those fall back to a decomposition on CPU.
    try:
        _, _, right_vectors = torch.linalg.svd(matrix, full_matrices=False)
    except NotImplementedError:
        if not cpu_fallback:
            raise
        _, _, right_vectors = torch.linalg.svd(matrix.cpu(), full_matrices=False)
        right_vectors = right_vectors.to(matrix.device)
    return right_vectors


class EigenCAM(_CAM):
    r"""Extract a class-independent map using the first principal component of spatial features.

    Implements the projection from ["Eigen-CAM: Class Activation Map using Principal
    Components"](https://arxiv.org/abs/2008.00299). Spatial features are centered per channel before
    reduced SVD, following the [pytorch-grad-cam reference implementation](https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/svd_on_activations.py).
    For each sample, the centered spatial-by-channel matrix $A$ is projected onto its first right
    singular vector $v_1$, giving $L = ReLU(A v_1)$.

    SVD vectors have an arbitrary sign. Here, the largest-magnitude channel loading is made positive
    before projection. Repeated largest singular values can still yield different principal directions.
    No classifier weights, scores, backward pass or additional model forward are needed. ``class_idx``
    may be omitted; valid class indices are accepted for API compatibility but do not change the map.
    Callable output targets are unsupported because the map is not class-discriminative.

    Half-precision features are promoted to float32 for SVD and projection; float64 is preserved.
    Maps stay on the activation device. On MPS versions without SVD, only the decomposition runs on CPU.
    Constant spatial features produce zero maps, including when normalization is disabled.

    Example:
        ```python
        from torchcam.methods import EigenCAM
        with EigenCAM(model, target_layer="layer4") as extractor:
            with torch.inference_mode():
                model(input_tensor)
                cams = extractor()
        ```

    Args:
        model: input model
        target_layer: target layer or its name, or a list of layers or names
        input_shape: expected input shape excluding the batch dimension
        enable_hooks: whether hooks are enabled by default
        reshape_transform: optional transform mapping hooked features to spatial tensors
    """

    def _precheck(
        self,
        class_idx: int | list[int] | None,
        scores: Any = None,
        targets: OutputTarget | list[OutputTarget] | None = None,
    ) -> None:
        # Reuse the hook and optional class-index checks without requiring a class for this method.
        super()._precheck(0 if class_idx is None and targets is None else class_idx, scores, targets)

    @torch.no_grad()
    def _get_weights(self, class_idx: int | list[int], *_: Any, **kwargs: Any) -> list[Tensor]:  # noqa: ARG002
        """Compute each sample's first principal direction."""  # noqa: DOC201, DOC501
        weights = []
        for activation in cast(list[Tensor], self.hook_a):
            if activation.ndim < 3 or any(dim == 0 for dim in activation.shape):
                raise ValueError("EigenCAM requires non-empty spatial activations of shape (N, C, ...)")
            if not activation.is_floating_point() or not activation.isfinite().all():
                raise ValueError("EigenCAM requires finite floating-point activations")
            dtype = torch.float64 if activation.dtype == torch.float64 else torch.float32
            features = activation.to(dtype).flatten(2).transpose(1, 2)
            centered_features = features - features.mean(dim=1, keepdim=True)
            right_vectors = _svd_right_vectors(centered_features, cpu_fallback=features.device.type == "mps")
            direction = right_vectors[:, 0, :]
            pivot = direction.gather(1, direction.abs().argmax(dim=1, keepdim=True))
            weights.append(direction * pivot.sign())
        return weights

    @torch.no_grad()
    def compute_cams(
        self,
        class_idx: int | list[int] | None = None,
        scores: Any = None,
        normalized: bool = True,
        *,
        targets: OutputTarget | list[OutputTarget] | None = None,
        **kwargs: Any,
    ) -> list[Tensor]:
        """Project centered activations onto their first principal directions.

        Args:
            class_idx: optional class indices, ignored by this class-independent method
            scores: optional model output, ignored by this method
            normalized: min-max normalize each map to [0, 1] after ReLU
            targets: unsupported scalar output targets
            kwargs: keyword arguments passed to `_get_weights`

        Returns:
            detached maps of shape (N, ...), one per target layer, in float32 or float64
        """
        self._precheck(class_idx, scores, targets)
        weights = self._get_weights(cast(int | list[int], class_idx), scores, **kwargs)
        cams = []
        for weight, activation in zip(weights, cast(list[Tensor], self.hook_a), strict=True):
            features = activation.to(weight.dtype)
            centered_features = features - features.mean(dim=tuple(range(2, features.ndim)), keepdim=True)
            cam = (centered_features * weight[(...,) + (None,) * (features.ndim - 2)]).sum(dim=1).relu_()
            cams.append(self._normalize(cam) if normalized else cam)
        return cams


class CAM(_CAM):
    r"""Implements a class activation map extractor as described in ["Learning Deep Features for Discriminative
    Localization"](https://arxiv.org/pdf/1512.04150.pdf).

    The Class Activation Map (CAM) is defined for image classification models that have global pooling at the end
    of the visual feature extraction block. The localization map is computed as follows:

    $$
    L^{(c)}_{CAM}(x, y) = ReLU\Big(\sum\limits_k w_k^{(c)} A_k(x, y)\Big)
    $$

    where $A_k(x, y)$ is the activation of node $k$ in the target layer of the model at
    position $(x, y)$,
    and $w_k^{(c)}$ is the weight corresponding to class $c$ for unit $k$ in the fully
    connected layer.

    Example:
        ```python
        from torchvision.models import get_model, get_model_weights
        from torchcam.methods import CAM
        model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
        with CAM(model, 'layer4', 'fc') as cam_extractor:
            with torch.inference_mode(): out = model(input_tensor)
            cam = cam_extractor(class_idx=100)
        ```

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        fc_layer: either the fully connected layer itself or its name
        input_shape: shape of the expected input tensor excluding the batch dimension

    Raises:
        ValueError: if the argument is invalid
        TypeError: if the argument type is invalid
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | str | list[nn.Module | str] | None = None,
        fc_layer: nn.Module | str | None = None,
        input_shape: tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:
        if isinstance(target_layer, list) and len(target_layer) > 1:
            raise ValueError("base CAM does not support multiple target layers")

        super().__init__(model, target_layer, input_shape, **kwargs)

        if isinstance(fc_layer, str):
            fc_name = fc_layer
        # Find the location of the module
        elif isinstance(fc_layer, nn.Module):
            fc_name = self._resolve_layer_name(fc_layer)
        # If the layer is not specified, try automatic resolution
        elif fc_layer is None:
            lin_layers = [layer_name for layer_name, m in model.named_modules() if isinstance(m, nn.Linear)]
            # Warn the user of the choice
            if len(lin_layers) == 0:
                raise ValueError("unable to resolve `fc_layer` automatically, please specify its value.")
            if len(lin_layers) > 1:
                raise ValueError("This CAM method does not support multiple fully connected layers.")
            fc_name = lin_layers[0]
            logger.warning(f"no value was provided for `fc_layer`, thus set to '{fc_name}'.")
        else:
            raise TypeError("invalid argument type for `fc_layer`")
        # Softmax weight
        self._fc_weights = self.submodule_dict[fc_name].weight.data
        # squeeze to accomodate replacement by Conv1x1
        if self._fc_weights.ndim > 2:
            self._fc_weights = self._fc_weights.view(*self._fc_weights.shape[:2])

    @torch.no_grad()
    def _get_weights(
        self,
        class_idx: int | list[int],
        *_: Any,
    ) -> list[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""  # noqa: DOC201
        # Take the FC weights of the target class
        if isinstance(class_idx, int):
            return [self._fc_weights[class_idx, :].unsqueeze(0)]
        return [self._fc_weights[class_idx, :]]


class ScoreCAM(_CAM):
    r"""Implements a class activation map extractor as described in ["Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks"](https://arxiv.org/pdf/1910.01279.pdf).

    The localization map is computed as follows:

    $$
    L^{(c)}_{Score-CAM}(x, y) = ReLU\Big(\sum\limits_k w_k^{(c)} A_k(x, y)\Big)
    $$

    with the coefficient $w_k^{(c)}$ being defined as:

    $$
    w_k^{(c)} = softmax\Big(Y^{(c)}(M_k) - Y^{(c)}(X_b)\Big)_k
    $$

    where $A_k(x, y)$ is the activation of node $k$ in the target layer of the model at
    position $(x, y)$, $Y^{(c)}(X)$ is the model output score for class $c$ before softmax
    for input $X$, $X_b$ is a baseline image,
    and $M_k$ is defined as follows:

    $$
    M_k = \frac{U(A_k) - \min\limits_m U(A_m)}{\max\limits_m  U(A_m) - \min\limits_m  U(A_m)})
    \odot X_b
    $$

    where $\odot$ refers to the element-wise multiplication and $U$ is the upsampling operation.

    Example:
        ```python
        from torchvision.models import get_model, get_model_weights
        from torchcam.methods import ScoreCAM
        model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
        with ScoreCAM(model, 'layer4') as cam_extractor:
            with torch.inference_mode(): out = model(input_tensor)
            cam = cam_extractor(class_idx=100)
        ```

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        batch_size: batch size used to upsample masks and forward masked inputs
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | str | list[nn.Module | str] | None = None,
        batch_size: int = 32,
        input_shape: tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:
        super().__init__(model, target_layer, input_shape, **kwargs)

        # Input hook
        self.hook_handles.append(model.register_forward_pre_hook(self._store_input))
        self.bs = batch_size
        # Ensure ReLU is applied to CAM before normalization
        self._relu = True
        self._targets_supported = True

    def _store_input(self, _: nn.Module, input_: tuple[Any, ...]) -> None:
        """Store model input tensor."""
        if self._hooks_enabled:
            self._input = input_[0].detach().clone()

    def _prepare_masked_inputs(self, activation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Prepare activation masks and their sample indices.

        Returns:
            mask source, flattened sample indices and reusable input buffer
        """
        batch, channels = activation.shape[:2]
        sample_indices = torch.arange(batch).repeat_interleave(channels)
        return (
            activation.flatten(0, 1).unsqueeze(1),
            sample_indices,
            self._input.new_empty((min(self.bs, batch * channels), *self._input.shape[1:])),
        )

    def _masked_input_chunk(self, masks: Tensor, sample_indices: Tensor, buffer: Tensor, slice_: slice) -> Tensor:
        model_input = self._input if self._input.shape[0] == 1 else self._input[sample_indices[slice_]]
        chunk = buffer[: slice_.stop - slice_.start]
        masks = masks[slice_]
        if masks.shape[2:] != self._input.shape[2:]:
            masks = self._upsample(masks)
        return torch.mul(masks, model_input, out=chunk)

    def _upsample(self, activation: Tensor) -> Tensor:
        spatial_dims = self._input.ndim - 2
        interpolation_mode = "bilinear" if spatial_dims == 2 else "trilinear" if spatial_dims == 3 else "nearest"
        return F.interpolate(
            activation,
            self._input.shape[2:],
            mode=interpolation_mode,
            align_corners=None if interpolation_mode == "nearest" else False,
        )

    @staticmethod
    def _select_scores(
        output: Any,
        sample_indices: Tensor,
        class_idx: int | list[int] | None,
        targets: list[OutputTarget] | None,
    ) -> Tensor:
        if targets is not None:
            selected_targets = [targets[idx] for idx in sample_indices.tolist()]
            return _target_scores(output, selected_targets)

        if isinstance(class_idx, int):
            return cast(Tensor, output)[:, class_idx]
        indices = torch.tensor(class_idx, device=cast(Tensor, output).device)[sample_indices]
        return cast(Tensor, output).gather(1, indices.view(-1, 1)).squeeze(1)

    def _init_score_weights(
        self,
        activations: list[Tensor],
        class_idx: int | list[int] | None,
        targets: OutputTarget | list[OutputTarget] | None,
    ) -> tuple[list[Tensor], list[OutputTarget] | None, Tensor]:
        weights = [act.new_zeros(act.shape[0] * act.shape[1]) for act in activations]
        logits = self.model(self._input)
        target_fns = _resolve_targets(targets, activations[0].shape[0]) if targets is not None else None
        batch_indices = torch.arange(activations[0].shape[0])
        return weights, target_fns, self._select_scores(logits, batch_indices, class_idx, target_fns)

    def _score_input(
        self,
        input_tensor: Tensor,
        sample_indices: Tensor,
        class_idx: int | list[int] | None,
        targets: list[OutputTarget] | None,
    ) -> Tensor:
        return self._select_scores(self.model(input_tensor), sample_indices, class_idx, targets)

    @torch.no_grad()
    def _get_score_weights(
        self,
        activations: list[Tensor],
        class_idx: int | list[int] | None,
        targets: OutputTarget | list[OutputTarget] | None = None,
    ) -> list[Tensor]:
        weights, target_fns, baseline_scores = self._init_score_weights(activations, class_idx, targets)

        for activation, weight in zip(activations, weights, strict=True):
            masks, sample_indices, buffer = self._prepare_masked_inputs(activation)
            # Process by chunk (GPU RAM limitation)
            for idx_ in range(math.ceil(weight.numel() / self.bs)):
                slice_ = slice(idx_ * self.bs, min((idx_ + 1) * self.bs, weight.numel()))
                chunk_indices = sample_indices[slice_]
                chunk_scores = self._score_input(
                    self._masked_input_chunk(masks, sample_indices, buffer, slice_),
                    chunk_indices,
                    class_idx,
                    target_fns,
                )
                weight[slice_] = chunk_scores - baseline_scores[chunk_indices]

        # Reshape the weights (N, C)
        return [
            torch.softmax(weight.view(*activation.shape[:2]), -1)
            for weight, activation in zip(weights, activations, strict=True)
        ]

    @torch.no_grad()
    def _get_weights(
        self,
        class_idx: int | list[int] | None,
        *_: Any,
        targets: OutputTarget | list[OutputTarget] | None = None,
    ) -> list[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""  # noqa: DOC201
        hook_a = cast(list[Tensor], self.hook_a)

        # Normalize the activation
        # (N, C, H', W')
        activations = [self._normalize(act.clone(), act.ndim - 2) for act in hook_a]

        with self._hooks_off(), self._eval_mode():
            return self._get_score_weights(activations, class_idx, targets)

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(batch_size={self.bs})"


class SSCAM(ScoreCAM):
    r"""Implements a class activation map extractor as described in ["SS-CAM: Smoothed Score-CAM for
    Sharper Visual Feature Localization"](https://arxiv.org/pdf/2006.14255.pdf).

    The localization map is computed as follows:

    $$
    L^{(c)}_{SS-CAM}(x, y) = ReLU\Big(\sum\limits_k w_k^{(c)} A_k(x, y)\Big)
    $$

    with the coefficient $w_k^{(c)}$ being defined as:

    $$
    w_k^{(c)} = softmax\Big(\frac{1}{N} \sum\limits_{i=1}^N (Y^{(c)}(\hat{M_k}) - Y^{(c)}(X_b))\Big)_k
    $$

    where $N$ is the number of samples used to smooth the weights,
    $A_k(x, y)$ is the activation of node $k$ in the target layer of the model at
    position $(x, y)$, $Y^{(c)}(X)$ is the model output score for class $c$ before softmax
    for input $X$, $X_b$ is a baseline image,
    and $M_k$ is defined as follows:

    $$
    \hat{M_k} = \Bigg(\frac{U(A_k) - \min\limits_m U(A_m)}{\max\limits_m  U(A_m) - \min\limits_m  U(A_m)} +
    \delta\Bigg) \odot X_b
    $$

    where $\odot$ refers to the element-wise multiplication, $U$ is the upsampling operation,
    $\delta \sim \mathcal{N}(0, \sigma^2)$ is the random noise that follows a 0-mean gaussian distribution
    with a standard deviation of $\sigma$.

    Example:
        ```python
        from torchvision.models import get_model, get_model_weights
        from torchcam.methods import SSCAM
        model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
        with SSCAM(model, 'layer4') as cam_extractor:
            with torch.inference_mode(): out = model(input_tensor)
            cam = cam_extractor(class_idx=100)
        ```

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
        target_layer: nn.Module | str | list[nn.Module | str] | None = None,
        batch_size: int = 32,
        num_samples: int = 35,
        std: float = 2.0,
        input_shape: tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:
        super().__init__(model, target_layer, batch_size, input_shape, **kwargs)

        self.num_samples = num_samples
        self.std = std
        self._distrib = torch.distributions.normal.Normal(0, self.std)

    @torch.no_grad()
    def _get_score_weights(
        self,
        activations: list[Tensor],
        class_idx: int | list[int] | None,
        targets: OutputTarget | list[OutputTarget] | None = None,
    ) -> list[Tensor]:
        # Noise must still be sampled at input resolution, in the same order.
        activations = [self._upsample(act) for act in activations]
        weights, target_fns, baseline_scores = self._init_score_weights(activations, class_idx, targets)

        for activation, weight in zip(activations, weights, strict=True):
            # Add noise
            for _ in range(self.num_samples):
                noise = self._distrib.sample(activation.size()).to(device=activation.device)
                masks, sample_indices, buffer = self._prepare_masked_inputs(activation + noise)

                # Process by chunk (GPU RAM limitation)
                for idx_ in range(math.ceil(weight.numel() / self.bs)):
                    slice_ = slice(idx_ * self.bs, min((idx_ + 1) * self.bs, weight.numel()))
                    chunk_indices = sample_indices[slice_]
                    chunk_scores = self._score_input(
                        self._masked_input_chunk(masks, sample_indices, buffer, slice_),
                        chunk_indices,
                        class_idx,
                        target_fns,
                    )
                    weight[slice_] += chunk_scores - baseline_scores[chunk_indices]

        # Reshape the weights (N, C)
        return [
            torch.softmax(weight.div_(self.num_samples).view(*activation.shape[:2]), -1)
            for weight, activation in zip(weights, activations, strict=True)
        ]

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(batch_size={self.bs}, num_samples={self.num_samples}, std={self.std})"


class ISCAM(ScoreCAM):
    r"""Implements a class activation map extractor as described in ["IS-CAM: Integrated Score-CAM for axiomatic-based
    explanations"](https://arxiv.org/pdf/2010.03023.pdf).

    The localization map is computed as follows:

    $$
    L^{(c)}_{ISS-CAM}(x, y) = ReLU\Big(\sum\limits_k w_k^{(c)} A_k(x, y)\Big)
    $$

    with the coefficient $w_k^{(c)}$ being defined as:

    $$
    w_k^{(c)} = softmax\Bigg(\frac{1}{N} \sum\limits_{i=1}^N
    \Big(Y^{(c)}(M_i) - Y^{(c)}(X_b)\Big)\Bigg)_k
    $$

    where $N$ is the number of samples used to smooth the weights,
    $A_k(x, y)$ is the activation of node $k$ in the target layer of the model at
    position $(x, y)$, $Y^{(c)}(X)$ is the model output score for class $c$ before softmax
    for input $X$, $X_b$ is a baseline image,
    and $M_i$ is defined as follows:

    $$
    M_i = \sum\limits_{j=0}^{i-1} \frac{j}{N}
    \frac{U(A_k) - \min\limits_m U(A_m)}{\max\limits_m  U(A_m) - \min\limits_m  U(A_m)} \odot X_b
    $$

    where $\odot$ refers to the element-wise multiplication, $U$ is the upsampling operation.

    Example:
        ```python
        from torchvision.models import get_model, get_model_weights
        from torchcam.methods import ISCAM
        model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
        with ISCAM(model, 'layer4') as cam_extractor:
            with torch.inference_mode(): out = model(input_tensor)
            cam = cam_extractor(class_idx=100)
        ```

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
        target_layer: nn.Module | str | list[nn.Module | str] | None = None,
        batch_size: int = 32,
        num_samples: int = 10,
        input_shape: tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:
        super().__init__(model, target_layer, batch_size, input_shape, **kwargs)

        self.num_samples = num_samples

    @torch.no_grad()
    def _get_score_weights(
        self,
        activations: list[Tensor],
        class_idx: int | list[int] | None,
        targets: OutputTarget | list[OutputTarget] | None = None,
    ) -> list[Tensor]:
        # Preserve integration/model-call order without repeatedly interpolating masks.
        activations = [self._upsample(act) for act in activations]
        prepared_inputs = [self._prepare_masked_inputs(act) for act in activations]

        weights, target_fns, baseline_scores = self._init_score_weights(activations, class_idx, targets)

        for (masks, sample_indices, buffer), weight in zip(prepared_inputs, weights, strict=True):
            coeff = 0.0
            # Process by chunk (GPU RAM limitation)
            for sidx in range(self.num_samples):
                coeff += (sidx + 1) / self.num_samples

                # Process by chunk (GPU RAM limitation)
                for idx_ in range(math.ceil(weight.numel() / self.bs)):
                    slice_ = slice(idx_ * self.bs, min((idx_ + 1) * self.bs, weight.numel()))
                    chunk_indices = sample_indices[slice_]
                    chunk_scores = self._score_input(
                        coeff * self._masked_input_chunk(masks, sample_indices, buffer, slice_),
                        chunk_indices,
                        class_idx,
                        target_fns,
                    )
                    weight[slice_] += chunk_scores - baseline_scores[chunk_indices]

        # Reshape the weights (N, C)
        return [
            torch.softmax(weight.div_(self.num_samples).view(*activation.shape[:2]), -1)
            for weight, activation in zip(weights, activations, strict=True)
        ]

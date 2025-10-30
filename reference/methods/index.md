# Interpretability methods

## Class activation map

The class activation map gives you the importance of each region of a feature map on a model's output. More specifically, a class activation map is relative to:

- the layer at which it is computed (e.g. the N-th layer of your model)
- the model's classification output (e.g. the raw logits of the model)
- the class index to focus on

With TorchCAM, the target layer is selected when you create your CAM extractor. You will need to pass the model logits to the extractor and a class index for it to do its magic!

## Activation-based methods

Methods related to activation-based class activation maps.

### CAM

```python
CAM(model: Module, target_layer: Module | str | list[Module | str] | None = None, fc_layer: Module | str | None = None, input_shape: tuple[int, ...] = (3, 224, 224), **kwargs: Any)
```

Implements a class activation map extractor as described in ["Learning Deep Features for Discriminative Localization"](https://arxiv.org/pdf/1512.04150.pdf).

The Class Activation Map (CAM) is defined for image classification models that have global pooling at the end of the visual feature extraction block. The localization map is computed as follows:

[ L^{(c)}\_{CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big) ]

where (A_k(x, y)) is the activation of node (k) in the target layer of the model at position ((x, y)), and (w_k^{(c)}) is the weight corresponding to class (c) for unit (k) in the fully connected layer.

Example

```python
from torchvision.models import get_model, get_model_weights
from torchcam.methods import CAM
model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
with CAM(model, 'layer4', 'fc') as cam_extractor:
    with torch.inference_mode(): out = model(input_tensor)
    cam = cam_extractor(class_idx=100)
```

| PARAMETER      | DESCRIPTION                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `model`        | input model **TYPE:** `Module`                                                                                            |
| `target_layer` | either the target layer itself or its name, or a list of those **TYPE:** \`Module                                         |
| `fc_layer`     | either the fully connected layer itself or its name **TYPE:** \`Module                                                    |
| `input_shape`  | shape of the expected input tensor excluding the batch dimension **TYPE:** `tuple[int, ...]` **DEFAULT:** `(3, 224, 224)` |

| RAISES       | DESCRIPTION                     |
| ------------ | ------------------------------- |
| `ValueError` | if the argument is invalid      |
| `TypeError`  | if the argument type is invalid |

Source code in `torchcam/methods/activation.py`

```python
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
```

### torchcam.methods.CAM

```python
CAM(model: Module, target_layer: Module | str | list[Module | str] | None = None, fc_layer: Module | str | None = None, input_shape: tuple[int, ...] = (3, 224, 224), **kwargs: Any)
```

Implements a class activation map extractor as described in ["Learning Deep Features for Discriminative Localization"](https://arxiv.org/pdf/1512.04150.pdf).

The Class Activation Map (CAM) is defined for image classification models that have global pooling at the end of the visual feature extraction block. The localization map is computed as follows:

[ L^{(c)}\_{CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big) ]

where (A_k(x, y)) is the activation of node (k) in the target layer of the model at position ((x, y)), and (w_k^{(c)}) is the weight corresponding to class (c) for unit (k) in the fully connected layer.

Example

```python
from torchvision.models import get_model, get_model_weights
from torchcam.methods import CAM
model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
with CAM(model, 'layer4', 'fc') as cam_extractor:
    with torch.inference_mode(): out = model(input_tensor)
    cam = cam_extractor(class_idx=100)
```

| PARAMETER      | DESCRIPTION                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `model`        | input model **TYPE:** `Module`                                                                                            |
| `target_layer` | either the target layer itself or its name, or a list of those **TYPE:** \`Module                                         |
| `fc_layer`     | either the fully connected layer itself or its name **TYPE:** \`Module                                                    |
| `input_shape`  | shape of the expected input tensor excluding the batch dimension **TYPE:** `tuple[int, ...]` **DEFAULT:** `(3, 224, 224)` |

| RAISES       | DESCRIPTION                     |
| ------------ | ------------------------------- |
| `ValueError` | if the argument is invalid      |
| `TypeError`  | if the argument type is invalid |

Source code in `torchcam/methods/activation.py`

```python
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
```

### torchcam.methods.ScoreCAM

```python
ScoreCAM(model: Module, target_layer: Module | str | list[Module | str] | None = None, batch_size: int = 32, input_shape: tuple[int, ...] = (3, 224, 224), **kwargs: Any)
```

Implements a class activation map extractor as described in ["Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks"](https://arxiv.org/pdf/1910.01279.pdf).

The localization map is computed as follows:

[ L^{(c)}\_{Score-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big) ]

with the coefficient (w_k^{(c)}) being defined as:

[ w_k^{(c)} = softmax\\Big(Y^{(c)}(M_k) - Y^{(c)}(X_b)\\Big)\_k ]

where (A_k(x, y)) is the activation of node (k) in the target layer of the model at position ((x, y)), (Y^{(c)}(X)) is the model output score for class (c) before softmax for input (X), (X_b) is a baseline image, and (M_k) is defined as follows:

[ M_k = \\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m U(A_m) - \\min\\limits_m U(A_m)}) \\odot X_b ]

where (\\odot) refers to the element-wise multiplication and (U) is the upsampling operation.

Example

```python
from torchvision.models import get_model, get_model_weights
from torchcam.methods import ScoreCAM
model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
with ScoreCAM(model, 'layer4') as cam_extractor:
    with torch.inference_mode(): out = model(input_tensor)
    cam = cam_extractor(class_idx=100)
```

| PARAMETER      | DESCRIPTION                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `model`        | input model **TYPE:** `Module`                                                                                            |
| `target_layer` | either the target layer itself or its name, or a list of those **TYPE:** \`Module                                         |
| `batch_size`   | batch size used to forward masked inputs **TYPE:** `int` **DEFAULT:** `32`                                                |
| `input_shape`  | shape of the expected input tensor excluding the batch dimension **TYPE:** `tuple[int, ...]` **DEFAULT:** `(3, 224, 224)` |

Source code in `torchcam/methods/activation.py`

```python
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
    self.hook_handles.append(model.register_forward_pre_hook(self._store_input))  # type: ignore[arg-type]
    self.bs = batch_size
    # Ensure ReLU is applied to CAM before normalization
    self._relu = True
```

### torchcam.methods.SSCAM

```python
SSCAM(model: Module, target_layer: Module | str | list[Module | str] | None = None, batch_size: int = 32, num_samples: int = 35, std: float = 2.0, input_shape: tuple[int, ...] = (3, 224, 224), **kwargs: Any)
```

Implements a class activation map extractor as described in ["SS-CAM: Smoothed Score-CAM for Sharper Visual Feature Localization"](https://arxiv.org/pdf/2006.14255.pdf).

The localization map is computed as follows:

[ L^{(c)}\_{SS-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big) ]

with the coefficient (w_k^{(c)}) being defined as:

[ w_k^{(c)} = softmax\\Big(\\frac{1}{N} \\sum\\limits\_{i=1}^N (Y^{(c)}(\\hat{M_k}) - Y^{(c)}(X_b))\\Big)\_k ]

where (N) is the number of samples used to smooth the weights, (A_k(x, y)) is the activation of node (k) in the target layer of the model at position ((x, y)), (Y^{(c)}(X)) is the model output score for class (c) before softmax for input (X), (X_b) is a baseline image, and (M_k) is defined as follows:

[ \\hat{M_k} = \\Bigg(\\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m U(A_m) - \\min\\limits_m U(A_m)} + \\delta\\Bigg) \\odot X_b ]

where (\\odot) refers to the element-wise multiplication, (U) is the upsampling operation, (\\delta \\sim \\mathcal{N}(0, \\sigma^2)) is the random noise that follows a 0-mean gaussian distribution with a standard deviation of (\\sigma).

Example

```python
from torchvision.models import get_model, get_model_weights
from torchcam.methods import SSCAM
model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
with SSCAM(model, 'layer4') as cam_extractor:
    with torch.inference_mode(): out = model(input_tensor)
    cam = cam_extractor(class_idx=100)
```

| PARAMETER      | DESCRIPTION                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `model`        | input model **TYPE:** `Module`                                                                                            |
| `target_layer` | either the target layer itself or its name, or a list of those **TYPE:** \`Module                                         |
| `batch_size`   | batch size used to forward masked inputs **TYPE:** `int` **DEFAULT:** `32`                                                |
| `num_samples`  | number of noisy samples used for weight computation **TYPE:** `int` **DEFAULT:** `35`                                     |
| `std`          | standard deviation of the noise added to the normalized activation **TYPE:** `float` **DEFAULT:** `2.0`                   |
| `input_shape`  | shape of the expected input tensor excluding the batch dimension **TYPE:** `tuple[int, ...]` **DEFAULT:** `(3, 224, 224)` |

Source code in `torchcam/methods/activation.py`

```python
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
    self._distrib = torch.distributions.normal.Normal(0, self.std)  # ty: ignore[unresolved-attribute]
```

### torchcam.methods.ISCAM

```python
ISCAM(model: Module, target_layer: Module | str | list[Module | str] | None = None, batch_size: int = 32, num_samples: int = 10, input_shape: tuple[int, ...] = (3, 224, 224), **kwargs: Any)
```

Implements a class activation map extractor as described in ["IS-CAM: Integrated Score-CAM for axiomatic-based explanations"](https://arxiv.org/pdf/2010.03023.pdf).

The localization map is computed as follows:

[ L^{(c)}\_{ISS-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big) ]

with the coefficient (w_k^{(c)}) being defined as:

[ w_k^{(c)} = softmax\\Bigg(\\frac{1}{N} \\sum\\limits\_{i=1}^N \\Big(Y^{(c)}(M_i) - Y^{(c)}(X_b)\\Big)\\Bigg)\_k ]

where (N) is the number of samples used to smooth the weights, (A_k(x, y)) is the activation of node (k) in the target layer of the model at position ((x, y)), (Y^{(c)}(X)) is the model output score for class (c) before softmax for input (X), (X_b) is a baseline image, and (M_i) is defined as follows:

[ M_i = \\sum\\limits\_{j=0}^{i-1} \\frac{j}{N} \\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m U(A_m) - \\min\\limits_m U(A_m)} \\odot X_b ]

where (\\odot) refers to the element-wise multiplication, (U) is the upsampling operation.

Example

```python
from torchvision.models import get_model, get_model_weights
from torchcam.methods import ISCAM
model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
with ISCAM(model, 'layer4') as cam_extractor:
    with torch.inference_mode(): out = model(input_tensor)
    cam = cam_extractor(class_idx=100)
```

| PARAMETER      | DESCRIPTION                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `model`        | input model **TYPE:** `Module`                                                                                            |
| `target_layer` | either the target layer itself or its name, or a list of those **TYPE:** \`Module                                         |
| `batch_size`   | batch size used to forward masked inputs **TYPE:** `int` **DEFAULT:** `32`                                                |
| `num_samples`  | number of noisy samples used for weight computation **TYPE:** `int` **DEFAULT:** `10`                                     |
| `input_shape`  | shape of the expected input tensor excluding the batch dimension **TYPE:** `tuple[int, ...]` **DEFAULT:** `(3, 224, 224)` |

Source code in `torchcam/methods/activation.py`

```python
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
```

## Gradient-based methods

Methods related to gradient-based class activation maps.

### torchcam.methods.GradCAM

```python
GradCAM(model: Module, target_layer: Module | str | list[Module | str] | None = None, input_shape: tuple[int, ...] = (3, 224, 224), **kwargs: Any)
```

Implements a class activation map extractor as described in ["Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"](https://arxiv.org/pdf/1610.02391.pdf).

The localization map is computed as follows:

[ L^{(c)}\_{Grad-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big) ]

with the coefficient (w_k^{(c)}) being defined as:

[ w_k^{(c)} = \\frac{1}{H \\cdot W} \\sum\\limits\_{i=1}^H \\sum\\limits\_{j=1}^W \\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)} ]

where (A_k(x, y)) is the activation of node (k) in the target layer of the model at position ((x, y)), and (Y^{(c)}) is the model output score for class (c) before softmax.

Example

```python
from torchvision.models import get_model, get_model_weights
from torchcam.methods import GradCAM
model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
with GradCAM(model, 'layer4') as cam_extractor:
    scores = model(input_tensor)
    cam = cam_extractor(class_idx=100, scores=scores)
```

| PARAMETER      | DESCRIPTION                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `model`        | input model **TYPE:** `Module`                                                                                            |
| `target_layer` | either the target layer itself or its name, or a list of those **TYPE:** \`Module                                         |
| `input_shape`  | shape of the expected input tensor excluding the batch dimension **TYPE:** `tuple[int, ...]` **DEFAULT:** `(3, 224, 224)` |

Source code in `torchcam/methods/gradient.py`

```python
def __init__(
    self,
    model: nn.Module,
    target_layer: nn.Module | str | list[nn.Module | str] | None = None,
    input_shape: tuple[int, ...] = (3, 224, 224),
    **kwargs: Any,
) -> None:
    super().__init__(model, target_layer, input_shape, **kwargs)
    # Ensure ReLU is applied before normalization
    self._relu = True
    # Model output is used by the extractor
    self._score_used = True
    for idx, name in enumerate(self.target_names):
        # Trick to avoid issues with inplace operations cf. https://github.com/pytorch/pytorch/issues/61519
        self.hook_handles.append(self.submodule_dict[name].register_forward_hook(partial(self._hook_g, idx=idx)))
```

### torchcam.methods.GradCAMpp

```python
GradCAMpp(model: Module, target_layer: Module | str | list[Module | str] | None = None, input_shape: tuple[int, ...] = (3, 224, 224), **kwargs: Any)
```

Implements a class activation map extractor as described in ["Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"](https://arxiv.org/pdf/1710.11063.pdf).

The localization map is computed as follows:

[ L^{(c)}\_{Grad-CAM++}(x, y) = \\sum\\limits_k w_k^{(c)} A_k(x, y) ]

with the coefficient (w_k^{(c)}) being defined as:

[ w_k^{(c)} = \\sum\\limits\_{i=1}^H \\sum\\limits\_{j=1}^W \\alpha_k^{(c)}(i, j) \\cdot ReLU\\Big(\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}\\Big) ]

where (A_k(x, y)) is the activation of node (k) in the target layer of the model at position ((x, y)), (Y^{(c)}) is the model output score for class (c) before softmax, and (\\alpha_k^{(c)}(i, j)) being defined as:

[ \\alpha_k^{(c)}(i, j) = \\frac{1}{\\sum\\limits\_{i, j} \\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}} = \\frac{\\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2}}{2 \\cdot \\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2} + \\sum\\limits\_{a,b} A_k (a,b) \\cdot \\frac{\\partial^3 Y^{(c)}}{(\\partial A_k(i,j))^3}} ]

if (\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)} = 1) else (0).

Example

```python
from torchvision.models import get_model, get_model_weights
from torchcam.methods import GradCAMpp
model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
with GradCAMpp(model, 'layer4') as cam_extractor:
    scores = model(input_tensor)
    cam = cam_extractor(class_idx=100, scores=scores)
```

| PARAMETER      | DESCRIPTION                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `model`        | input model **TYPE:** `Module`                                                                                            |
| `target_layer` | either the target layer itself or its name, or a list of those **TYPE:** \`Module                                         |
| `input_shape`  | shape of the expected input tensor excluding the batch dimension **TYPE:** `tuple[int, ...]` **DEFAULT:** `(3, 224, 224)` |

Source code in `torchcam/methods/gradient.py`

```python
def __init__(
    self,
    model: nn.Module,
    target_layer: nn.Module | str | list[nn.Module | str] | None = None,
    input_shape: tuple[int, ...] = (3, 224, 224),
    **kwargs: Any,
) -> None:
    super().__init__(model, target_layer, input_shape, **kwargs)
    # Ensure ReLU is applied before normalization
    self._relu = True
    # Model output is used by the extractor
    self._score_used = True
    for idx, name in enumerate(self.target_names):
        # Trick to avoid issues with inplace operations cf. https://github.com/pytorch/pytorch/issues/61519
        self.hook_handles.append(self.submodule_dict[name].register_forward_hook(partial(self._hook_g, idx=idx)))
```

### torchcam.methods.SmoothGradCAMpp

```python
SmoothGradCAMpp(model: Module, target_layer: Module | str | list[Module | str] | None = None, num_samples: int = 4, std: float = 0.3, input_shape: tuple[int, ...] = (3, 224, 224), **kwargs: Any)
```

Implements a class activation map extractor as described in ["Smooth Grad-CAM++: An Enhanced Inference Level Visualization Technique for Deep Convolutional Neural Network Models"](https://arxiv.org/pdf/1908.01224.pdf) with a personal correction to the paper (alpha coefficient numerator).

The localization map is computed as follows:

[ L^{(c)}\_{Smooth Grad-CAM++}(x, y) = \\sum\\limits_k w_k^{(c)} A_k(x, y) ]

with the coefficient (w_k^{(c)}) being defined as:

[ w_k^{(c)} = \\sum\\limits\_{i=1}^H \\sum\\limits\_{j=1}^W \\alpha_k^{(c)}(i, j) \\cdot ReLU\\Big(\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}\\Big) ]

where (A_k(x, y)) is the activation of node (k) in the target layer of the model at position ((x, y)), (Y^{(c)}) is the model output score for class (c) before softmax, and (\\alpha_k^{(c)}(i, j)) being defined as:

\[ \\alpha_k^{(c)}(i, j) = \\frac{\\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2}}{2 \\cdot \\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2} + \\sum\\limits\_{a,b} A_k (a,b) \\cdot \\frac{\\partial^3 Y^{(c)}}{(\\partial A_k(i,j))^3}} = \\frac{\\frac{1}{n} \\sum\\limits\_{m=1}^n D^{(c, 2)}_k(i, j)}{ \\frac{2}{n} \\sum\\limits_{m=1}^n D^{(c, 2)}_k(i, j) + \\sum\\limits_{a,b} A_k (a,b) \\cdot \\frac{1}{n} \\sum\\limits\_{m=1}^n D^{(c, 3)}\_k(i, j)} \]

if (\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)} = 1) else (0). Here (D^{(c, p)}\_k(i, j)) refers to the p-th partial derivative of the class score of class (c) relatively to the activation in layer (k) at position ((i, j)), and (n) is the number of samples used to get the gradient estimate.

Please note the difference in the numerator of (\\alpha_k^{(c)}(i, j)), which is actually (\\frac{1}{n} \\sum\\limits\_{k=1}^n D^{(c, 1)}\_k(i,j)) in the paper.

Example

```python
from torchvision.models import get_model, get_model_weights
from torchcam.methods import SmoothGradCAMpp
model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
with SmoothGradCAMpp(model, 'layer4') as cam_extractor:
    scores = model(input_tensor)
    cam = cam_extractor(class_idx=100)
```

| PARAMETER      | DESCRIPTION                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `model`        | input model **TYPE:** `Module`                                                                                            |
| `target_layer` | either the target layer itself or its name, or a list of those **TYPE:** \`Module                                         |
| `num_samples`  | number of samples to use for smoothing **TYPE:** `int` **DEFAULT:** `4`                                                   |
| `std`          | standard deviation of the noise **TYPE:** `float` **DEFAULT:** `0.3`                                                      |
| `input_shape`  | shape of the expected input tensor excluding the batch dimension **TYPE:** `tuple[int, ...]` **DEFAULT:** `(3, 224, 224)` |

Source code in `torchcam/methods/gradient.py`

```python
def __init__(
    self,
    model: nn.Module,
    target_layer: nn.Module | str | list[nn.Module | str] | None = None,
    num_samples: int = 4,
    std: float = 0.3,
    input_shape: tuple[int, ...] = (3, 224, 224),
    **kwargs: Any,
) -> None:
    super().__init__(model, target_layer, input_shape, **kwargs)
    # Model scores is not used by the extractor
    self._score_used = False

    # Input hook
    self.hook_handles.append(model.register_forward_pre_hook(self._store_input))  # type: ignore[arg-type]
    # Noise distribution
    self.num_samples = num_samples
    self.std = std
    self._distrib = torch.distributions.normal.Normal(0, self.std)  # ty: ignore[unresolved-attribute]
    # Specific input hook updater
    self._ihook_enabled = True
```

### torchcam.methods.XGradCAM

```python
XGradCAM(model: Module, target_layer: Module | str | list[Module | str] | None = None, input_shape: tuple[int, ...] = (3, 224, 224), **kwargs: Any)
```

Implements a class activation map extractor as described in ["Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs"](https://arxiv.org/pdf/2008.02312.pdf).

The localization map is computed as follows:

[ L^{(c)}\_{XGrad-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big) ]

with the coefficient (w_k^{(c)}) being defined as:

[ w_k^{(c)} = \\sum\\limits\_{i=1}^H \\sum\\limits\_{j=1}^W \\Big( \\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)} \\cdot \\frac{A_k(i, j)}{\\sum\\limits\_{m=1}^H \\sum\\limits\_{n=1}^W A_k(m, n)} \\Big) ]

where (A_k(x, y)) is the activation of node (k) in the target layer of the model at position ((x, y)), and (Y^{(c)}) is the model output score for class (c) before softmax.

Example

```python
from torchvision.models import get_model, get_model_weights
from torchcam.methods import XGradCAM
model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
with XGradCAM(model, 'layer4') as cam_extractor:
    scores = model(input_tensor)
    cam = cam_extractor(class_idx=100, scores=scores)
```

| PARAMETER      | DESCRIPTION                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `model`        | input model **TYPE:** `Module`                                                                                            |
| `target_layer` | either the target layer itself or its name, or a list of those **TYPE:** \`Module                                         |
| `input_shape`  | shape of the expected input tensor excluding the batch dimension **TYPE:** `tuple[int, ...]` **DEFAULT:** `(3, 224, 224)` |

Source code in `torchcam/methods/gradient.py`

```python
def __init__(
    self,
    model: nn.Module,
    target_layer: nn.Module | str | list[nn.Module | str] | None = None,
    input_shape: tuple[int, ...] = (3, 224, 224),
    **kwargs: Any,
) -> None:
    super().__init__(model, target_layer, input_shape, **kwargs)
    # Ensure ReLU is applied before normalization
    self._relu = True
    # Model output is used by the extractor
    self._score_used = True
    for idx, name in enumerate(self.target_names):
        # Trick to avoid issues with inplace operations cf. https://github.com/pytorch/pytorch/issues/61519
        self.hook_handles.append(self.submodule_dict[name].register_forward_hook(partial(self._hook_g, idx=idx)))
```

### torchcam.methods.LayerCAM

```python
LayerCAM(model: Module, target_layer: Module | str | list[Module | str] | None = None, input_shape: tuple[int, ...] = (3, 224, 224), **kwargs: Any)
```

Implements a class activation map extractor as described in ["LayerCAM: Exploring Hierarchical Class Activation Maps for Localization"](http://mmcheng.net/mftp/Papers/21TIP_LayerCAM.pdf).

The localization map is computed as follows:

[ L^{(c)}\_{Layer-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)}(x, y) \\cdot A_k(x, y)\\Big) ]

with the coefficient (w_k^{(c)}(x, y)) being defined as:

[ w_k^{(c)}(x, y) = ReLU\\Big(\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}(x, y)\\Big) ]

where (A_k(x, y)) is the activation of node (k) in the target layer of the model at position ((x, y)), and (Y^{(c)}) is the model output score for class (c) before softmax.

Example

```python
from torchvision.models import get_model, get_model_weights
from torchcam.methods import LayerCAM
model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
with LayerCAM(model, 'layer4') as cam_extractor:
    scores = model(input_tensor)
    cams = cam_extractor(class_idx=100, scores=scores)
    fused_cam = cam_extractor.fuse_cams(cams)
```

| PARAMETER      | DESCRIPTION                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `model`        | input model **TYPE:** `Module`                                                                                            |
| `target_layer` | either the target layer itself or its name, or a list of those **TYPE:** \`Module                                         |
| `input_shape`  | shape of the expected input tensor excluding the batch dimension **TYPE:** `tuple[int, ...]` **DEFAULT:** `(3, 224, 224)` |

Source code in `torchcam/methods/gradient.py`

```python
def __init__(
    self,
    model: nn.Module,
    target_layer: nn.Module | str | list[nn.Module | str] | None = None,
    input_shape: tuple[int, ...] = (3, 224, 224),
    **kwargs: Any,
) -> None:
    super().__init__(model, target_layer, input_shape, **kwargs)
    # Ensure ReLU is applied before normalization
    self._relu = True
    # Model output is used by the extractor
    self._score_used = True
    for idx, name in enumerate(self.target_names):
        # Trick to avoid issues with inplace operations cf. https://github.com/pytorch/pytorch/issues/61519
        self.hook_handles.append(self.submodule_dict[name].register_forward_hook(partial(self._hook_g, idx=idx)))
```

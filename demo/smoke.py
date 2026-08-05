# Copyright (C) 2021-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import gc
from threading import Lock

import torch
from torchvision.models import get_model

import app


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def hook_count(model):
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks) + len(module._backward_hooks)  # noqa: SLF001
        for module in model.modules()
    )


def main():
    check(app.compatible_methods("vit_b_16") == ("LeGrad",), "ViT compatibility changed")
    check("LeGrad" not in app.compatible_methods("resnet18"), "LeGrad must stay ViT-only")
    check("FinerCAM" in app.compatible_methods("resnet18"), "FinerCAM is missing")
    check("RefineCAM" in app.compatible_methods("resnet18"), "RefineCAM is missing")

    for model_name in app.MODEL_LABELS:
        model = get_model(model_name, weights=None)
        module_names = dict(model.named_modules())
        for method_name in app.compatible_methods(model_name):
            preset = app.target_layer_preset(model_name, method_name)
            for layer in preset:
                check(layer in module_names, f"Unknown {model_name} preset: {layer}")
            if method_name == "CAM":
                with app.build_extractor(model, method_name, list(preset)):
                    pass
        check(hook_count(model) == 0, f"Hooks leaked while checking {model_name}")
        del model
        gc.collect()

    model = get_model("resnet18", weights=None).train()
    module_modes = [module.training for module in model.modules()]
    parameter_flags = [parameter.requires_grad for parameter in model.parameters()]
    input_tensor = torch.rand(3, 64, 64)

    cam, _, _, _ = app.extract_cam(model, Lock(), input_tensor, "GradCAM", ["layer4"])
    check(tuple(cam.shape) == (1, 2, 2), "Unexpected GradCAM shape")
    check(hook_count(model) == 0, "Hooks leaked after successful extraction")
    check([module.training for module in model.modules()] == module_modes, "Module modes were not restored")
    check(
        [parameter.requires_grad for parameter in model.parameters()] == parameter_flags,
        "Parameter flags were not restored",
    )
    check(all(parameter.grad is None for parameter in model.parameters()), "Parameter gradients were not cleared")

    try:
        app.extract_cam(model, Lock(), input_tensor, "GradCAM", ["layer4"], class_idx=1000)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid class index did not fail")
    check(hook_count(model) == 0, "Hooks leaked after failed extraction")
    check([module.training for module in model.modules()] == module_modes, "Failure changed module modes")
    check(
        [parameter.requires_grad for parameter in model.parameters()] == parameter_flags,
        "Failure changed parameter flags",
    )
    check(all(parameter.grad is None for parameter in model.parameters()), "Failure left parameter gradients")


if __name__ == "__main__":
    main()

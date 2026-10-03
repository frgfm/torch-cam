# Copyright (C) 2021-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import gc
import struct
import zlib
from io import BytesIO
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import torch
from PIL import Image, PngImagePlugin, UnidentifiedImageError, features
from streamlit.testing.v1 import AppTest
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


def image_upload(image, image_format, **kwargs):
    source = BytesIO()
    source.name = "upload.png"
    image.save(source, format=image_format, **kwargs)
    return source


def check_image_decoding():
    for image_format, mode in (("JPEG", "L"), ("PNG", "RGBA")):
        image = Image.new(mode, (8, 4))
        source = image_upload(image, image_format)
        decoded = app.read_image(source)
        source.close()
        check(decoded.mode == "RGB", f"{image_format} was not converted to RGB")
        check(decoded.size == image.size, f"{image_format} dimensions changed")
        decoded.load()

    image = Image.new("RGB", (8, 4))
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise.
    oriented = app.read_image(image_upload(image, "JPEG", exif=exif))
    check(oriented.size == (4, 8), "EXIF orientation was not applied")

    # Filename filtering alone accepts these formats when renamed to .png.
    unsupported_formats = ["GIF", "BMP"]
    if features.check_codec("jpg_2000"):
        unsupported_formats.append("JPEG2000")
    for image_format in unsupported_formats:
        try:
            app.read_image(image_upload(image, image_format))
        except UnidentifiedImageError:
            pass
        else:
            raise AssertionError(f"Disguised {image_format} upload was accepted")

    try:
        app.read_image(BytesIO(b"\x89PNG\r\n\x1a\ntruncated"))
    except OSError:
        pass
    else:
        raise AssertionError("Truncated PNG upload was accepted")

    pixel_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = 24  # The 32-pixel image should trigger the warning, not the error.
        try:
            app.read_image(image_upload(image, "PNG"))
        except Image.DecompressionBombWarning:
            pass
        else:
            raise AssertionError("Oversized PNG upload was accepted")
    finally:
        Image.MAX_IMAGE_PIXELS = pixel_limit


def check_malformed_png_metadata():
    data = image_upload(Image.new("RGB", (1, 1)), "PNG").getvalue()
    end = data.index(b"IEND") - 4
    metadata_chunks = (
        (b"zTXt", b"key\0\1bad"),
        (b"gAMA", b""),
        (b"iCCP", b""),
        (b"eXIf", b"garbage"),
        (b"zTXt", b"key\0\0" + zlib.compress(b"a" * 33)),
    )
    with patch.object(PngImagePlugin, "MAX_TEXT_CHUNK", 32):
        for kind, payload in metadata_chunks:
            chunk = struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
            upload = data[:end] + chunk + data[end:]
            # Each file passes identification, then fails during loading or EXIF parsing.
            with Image.open(BytesIO(upload), formats=("JPEG", "PNG")) as image:
                check(image.format == "PNG", "Malformed fixture was not identified as PNG")
            try:
                app.read_image(BytesIO(upload))
            except OSError as exc:
                check(exc.__cause__ is not None, "Pillow parse error was not normalized")
            else:
                raise AssertionError(f"Malformed {kind!r} metadata was accepted")

            with patch("streamlit.file_uploader", return_value=BytesIO(upload)):
                ui = AppTest.from_file(Path(__file__).with_name("app.py")).run()
            check(len(ui.exception) == 0, "Malformed image crashed the demo")
            check(
                any("This image cannot be opened safely" in error.value for error in ui.error),
                "Malformed image did not show the safe-image error",
            )


def main():
    check_image_decoding()
    check_malformed_png_metadata()
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

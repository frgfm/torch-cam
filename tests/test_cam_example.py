from contextlib import nullcontext
from io import BytesIO
from unittest.mock import Mock

import pytest
from PIL import Image, UnidentifiedImageError

from scripts import cam_example


def test_load_image_from_local_path_and_url(tmp_path, monkeypatch):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), color="red").save(image_path)
    payload = image_path.read_bytes()

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://example.com/image.png"
        assert request.get_header("User-agent").startswith("TorchCAM")
        assert timeout == 5
        return BytesIO(payload)

    monkeypatch.setattr(cam_example, "urlopen", fake_urlopen)

    local_image = cam_example._load_image(str(image_path))
    url_image = cam_example._load_image("https://example.com/image.png")
    assert local_image.mode == url_image.mode == "RGB"
    assert local_image.size == url_image.size == (4, 4)


def test_load_image_propagates_read_errors(monkeypatch):
    response = Mock()
    response.read.side_effect = OSError("read failed")
    monkeypatch.setattr(cam_example, "urlopen", lambda *_args, **_kwargs: nullcontext(response))

    with pytest.raises(OSError, match="read failed"):
        cam_example._load_image("https://example.com/image.png")


def test_load_image_rejects_corrupt_payload(monkeypatch):
    monkeypatch.setattr(cam_example, "urlopen", lambda *_args, **_kwargs: BytesIO(b"not an image"))

    with pytest.raises(UnidentifiedImageError):
        cam_example._load_image("https://example.com/image.png")

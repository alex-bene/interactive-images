import requests
import torch
from PIL import Image

from interactive_images.models import ZoeDepth


def test_zoedepth():
    model = ZoeDepth("nyu")
    assert model.device.type == "cpu"
    assert model._device.type == "cpu"
    assert model.dtype == torch.float32
    assert model._dtype == torch.float32

    url = "https://shariqfarooq-zoedepth.hf.space/file=/home/user/app/examples/person_1.jpeg"
    image = Image.open(requests.get(url, stream=True, timeout=5).raw)
    image.thumbnail((512, 512))
    assert image.size == (341, 512)

    outputs = model(image)
    assert outputs[0]["predicted_depth"].shape == (512, 341)

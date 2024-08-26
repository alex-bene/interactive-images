"""A base class for depth estimation models"""

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from zoedepth_post_processing import post_process_depth_estimation_zoedepth


class DepthEstimationModel(torch.nn.Module):
    """Base class for depth estimation models"""

    def __init__(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | str = "float32",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.device = device
        self.dtype = dtype
        self.eval()

    @property
    def device(self):
        """Returns the device on which the model is running"""
        return self._device

    @device.setter
    def device(self, value: torch.device | str) -> torch.device:
        """Sets the device on which the model will run"""
        if isinstance(value, torch.device):
            self._device = value
        elif isinstance(value, str):
            self._device = torch.device(value)
        else:
            raise ValueError("Invalid device value. Must be a torch.device or a string.")

    @property
    def dtype(self) -> torch.dtype:
        """Returns the data type of the model's parameters and outputs"""
        return self._dtype

    @dtype.setter
    def dtype(self, value: torch.dtype | str):
        """Sets the data type of the model's parameters and outputs"""
        if isinstance(value, torch.dtype):
            if value != torch.float32 and self.device.type == "cpu":
                raise ValueError("CPU does not support non-float32 data types")
            self._dtype = value
        elif isinstance(value, str):
            if value != "float32" and self.device.type == "cpu":
                raise ValueError("CPU does not support non-float32 data types")
            self._dtype = getattr(torch, value)
        else:
            raise ValueError("Invalid dtype value. Must be a torch.dtype or a string.")

    @torch.inference_mode
    def __call__(self, x):
        raise NotImplementedError("Subclasses must implement the __call__ method")


class DepthEstimationModelHF(DepthEstimationModel):
    """Base class for depth estimation models using Hugging Face Transformers"""

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)

    @torch.inference_mode
    def __call__(self, images: list[Image.Image] | Image.Image):
        inputs = self.image_processor(images=images, return_tensors="pt").to(device=self.device, dtype=self.dtype)
        outputs = self.model(**inputs)

        return self.image_processor.post_process_depth_estimation(
            outputs, [images.size[::-1]] if isinstance(images, Image.Image) else [image.size[::-1] for image in images]
        )


class ZoeDepth(DepthEstimationModelHF):
    """Base class for depth estimation models using Hugging Face Transformers"""

    def __init__(self, model_variation: str, *args, **kwargs):
        if model_variation not in ["nyu", "kitti", "nyu-kitti"]:
            raise ValueError("Invalid model variation. Must be one of ['nyu', 'kitti', 'nyu-kitti']")

        super().__init__(model_name=f"Intel/zoedepth-{model_variation}", *args, **kwargs)

    @torch.inference_mode
    def __call__(self, images: list[Image.Image] | Image.Image):
        inputs = self.image_processor(images=images, return_tensors="pt").to(device=self.device, dtype=self.dtype)
        outputs = self.model(**inputs)
        outputs_flip = self.model(pixel_values=torch.flip(inputs.pixel_values, dims=[3]))

        return post_process_depth_estimation_zoedepth(
            outputs,
            [images.size[::-1]] if isinstance(images, Image.Image) else [image.size[::-1] for image in images],
            outputs_flip=outputs_flip,
        )


if __name__ == "__main__":
    import requests

    model = ZoeDepth("nyu")
    assert model.device.type == "cpu"
    assert model._device.type == "cpu"
    assert model.dtype == torch.float32
    assert model._dtype == torch.float32

    url = "https://shariqfarooq-zoedepth.hf.space/file=/home/user/app/examples/person_1.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)
    image.thumbnail((512, 512))
    assert image.size == (341, 512)

    outputs = model(image)
    assert outputs[0]["predicted_depth"].shape == outputs[0]["depth"].size == (512, 341)

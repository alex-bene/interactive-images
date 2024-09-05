"""A base class for depth estimation models."""

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .zoedepth_post_processing import post_process_depth_estimation_zoedepth


class DepthEstimationModel(torch.nn.Module):
    """Base class for depth estimation models."""

    def __init__(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | str = "float32",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.dtype = dtype
        self.eval()

    @property
    def device(self) -> torch.device:
        """Return the device on which the model is running."""
        return self._device

    @device.setter
    def device(self, value: torch.device | str) -> None:
        """Set the device on which the model will run."""
        if isinstance(value, torch.device):
            self._device = value
        elif isinstance(value, str):
            self._device = torch.device(value)
        else:
            msg = "Invalid device value. Must be a torch.device or a string."
            raise TypeError(msg)

    @property
    def dtype(self) -> torch.dtype:
        """Return the data type of the model's parameters and outputs."""
        return self._dtype

    @dtype.setter
    def dtype(self, value: torch.dtype | str) -> None:
        """Set the data type of the model's parameters and outputs."""
        if isinstance(value, torch.dtype):
            if value != torch.float32 and self.device.type == "cpu":
                msg = "CPU does not support non-float32 data types"
                raise ValueError(msg)
            self._dtype = value
        elif isinstance(value, str):
            if value != "float32" and self.device.type == "cpu":
                msg = "CPU does not support non-float32 data types"
                raise ValueError(msg)
            self._dtype = getattr(torch, value)
        else:
            msg = "Invalid dtype value. Must be a torch.dtype or a string."
            raise TypeError(msg)

    @torch.inference_mode
    def __call__(self):  # noqa: ANN204
        """Call the model to estimate depths from images."""
        msg = "Subclasses must implement the __call__ method"
        raise NotImplementedError(msg)


class DepthEstimationModelHF(DepthEstimationModel):
    """Base class for depth estimation models using Hugging Face Transformers."""

    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)

    @torch.inference_mode
    def __call__(self, images: list[Image.Image] | Image.Image) -> list[dict]:
        """Call the model to estimate depths from images."""
        inputs = self.image_processor(images=images, return_tensors="pt").to(device=self.device, dtype=self.dtype)
        outputs = self.model(**inputs)

        return self.image_processor.post_process_depth_estimation(
            outputs,
            [images.size[::-1]] if isinstance(images, Image.Image) else [image.size[::-1] for image in images],
        )


class ZoeDepth(DepthEstimationModelHF):
    """Base class for depth estimation models using Hugging Face Transformers."""

    def __init__(self, model_variation: str, *args, **kwargs) -> None:
        if model_variation not in ["nyu", "kitti", "nyu-kitti"]:
            msg = "Invalid model variation. Must be one of ['nyu', 'kitti', 'nyu-kitti']"
            raise ValueError(msg)

        super().__init__(*args, model_name=f"Intel/zoedepth-{model_variation}", **kwargs)

    @torch.inference_mode
    def __call__(self, images: list[Image.Image] | Image.Image) -> list[dict]:
        """Call the model to estimate depths from images."""
        inputs = self.image_processor(images=images, return_tensors="pt").to(device=self.device, dtype=self.dtype)
        outputs = self.model(**inputs)
        outputs_flip = self.model(pixel_values=torch.flip(inputs.pixel_values, dims=[3]))

        return post_process_depth_estimation_zoedepth(
            outputs,
            [images.size[::-1]] if isinstance(images, Image.Image) else [image.size[::-1] for image in images],
            outputs_flip=outputs_flip,
        )

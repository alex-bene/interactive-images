"""A collection of utility functions."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch3d import renderer
from pytorch3d.io import IO
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds


def save_textured_mesh(mesh: Meshes, filename: Path | str) -> None:
    """Save a mesh as `.obj` and it's texture of type TexturesVertex as `.pt`."""
    filename = Path(filename)
    IO().save_mesh(mesh, filename.with_suffix(".obj"))
    torch.save(mesh.textures.verts_features_list()[0], filename.with_suffix(".pt"))


def load_textured_mesh(filename: Path | str) -> Meshes:
    """Load an `.obj` mesh and it's texture of type TexturesVertex saved in `.pt` format."""
    filename = Path(filename)
    mesh = IO().load_mesh(filename.with_suffix(".obj"))
    textures = torch.load(filename.with_suffix(".pt"), weights_only=True)
    textures = TexturesVertex(verts_features=textures.unsqueeze(0))
    mesh.textures = textures

    return mesh


def save_pointcloud(pointcloud: Pointclouds, filename: Path | str) -> None:
    """Save a pointcloud as `.ply`."""
    IO().save_pointcloud(pointcloud, Path(filename).with_suffix(".ply"))


def load_pointcloud(filename: Path | str) -> Meshes:
    """Load an `.obj` mesh and it's texture of type TexturesVertex saved in `.pt` format."""
    return IO().load_pointcloud(Path(filename).with_suffix(".ply"))


def sph2cart(az: float, el: float, r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical coordinates (azimuth, elevation, radius) to Cartesian coordinates (x, y, z)."""
    # source https://github.com/numpy/numpy/issues/5228#issue-46746558
    rcos_theta = r * np.cos(np.radians(el))
    x = rcos_theta * np.cos(np.radians(az))
    y = rcos_theta * np.sin(np.radians(az))
    z = r * np.sin(np.radians(el))
    return np.round(x, 5), np.round(y, 5), np.round(z, 5)


def get_cameras_looking_at_points(
    camera_locations: torch.Tensor,
    image_size: torch.Tensor,
    look_at_points: torch.Tensor | None = None,
    focal_length: float | None = 0.7,
    dtype: torch.dtype | None = torch.float32,
) -> tuple[renderer.cameras.PerspectiveCameras, renderer.cameras.PerspectiveCameras | None]:
    """Create a batch of cameras looking at specific points."""
    # Extract device from input `camera_locations` tensor
    device = camera_locations.device

    number_of_cameras = camera_locations.shape[0]

    # `look_at_points` defaults to the center
    if look_at_points is None:
        look_at_points = torch.zeros_like(camera_locations)

    if not torch.is_tensor(focal_length):
        focal_length = torch.tensor([focal_length] * number_of_cameras).to(dtype=dtype, device=device)

    # Get camera rotation and translation
    R, T = renderer.look_at_view_transform(at=look_at_points, eye=camera_locations)

    image_size = image_size.unsqueeze(0).expand(number_of_cameras, -1)

    # Define Camera
    cameras = renderer.cameras.PerspectiveCameras(
        focal_length=focal_length,
        principal_point=image_size / 2,
        R=R,
        T=T,
        in_ndc=False,
        image_size=image_size.flip(-1),
        device=device,
    )

    return cameras[0] if number_of_cameras == 1 else cameras


def image_grid(imgs: Image.Image, rows: int, cols: int) -> Image.Image:
    """Create a grid of images."""
    if len(imgs) != rows * cols:
        msg = "Number of images must match the number of rows and columns."
        raise ValueError(msg)

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid

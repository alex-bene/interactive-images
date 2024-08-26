import numpy as np
import torch
from pytorch3d import renderer


def sph2cart(az, el, r):
    # source https://github.com/numpy/numpy/issues/5228#issue-46746558
    rcos_theta = r * np.cos(np.radians(el))
    x = rcos_theta * np.cos(np.radians(az))
    y = rcos_theta * np.sin(np.radians(az))
    z = r * np.sin(np.radians(el))
    return np.round(x, 5), np.round(y, 5), np.round(z, 5)


def get_cameras_looking_at_points(
    camera_locations, image_size, look_at_points=None, focal_length=0.7, dtype=torch.float32
):
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

    return (
        cameras[0],
        (
            renderer.camera_utils.join_cameras_as_batch([cameras[i] for i in range(1, number_of_cameras)])
            if number_of_cameras > 1
            else None
        ),
    )

"""A class that renders a 3D mesh from various cameras."""

import torch
from pytorch3d.renderer import (
    BlendParams,
    CamerasBase,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
)
from pytorch3d.structures import Meshes, join_meshes_as_batch


class MultiviewRenderer:
    """Class that renders a 3D mesh from various cameras."""

    def __init__(self, device: torch.device, render_cameras: CamerasBase) -> None:
        self.device = device
        BlendParams.background_color = (1.0, 1.0, 1.0)
        BlendParams.gamma = 0.001

        # Settings for rasterization
        raster_settings = RasterizationSettings(
            image_size=(512, 341),
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=50000,
        )

        # class Lighting(NamedTuple):  # pragma: no cover
        #     ambient: float = 0.8
        #     diffuse: float = 1.0
        #     fresnel: float = 0.0
        #     specular: float = 0.0
        #     roughness: float = 0.5
        #     facenormalsepsilon: float = 1e-6
        #     vertexnormalsepsilon: float = 1e-12

        # Setting the lights
        lights = PointLights(
            ambient_color=((0.7, 0.7, 0.7),),
            diffuse_color=((0.1, 0.1, 0.1),),
            specular_color=((0.1, 0.1, 0.1),),
            device=device,
            location=(torch.cat([render_camera.T for render_camera in render_cameras]).tolist()),
        )

        # Setting the renderer
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=render_cameras,
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=render_cameras,
                lights=lights,
            ),
        )

    def __call__(self, mesh: Meshes) -> torch.Tensor:
        """Render the mesh from multiple views."""
        return self.renderer(join_meshes_as_batch([mesh] * len(self.render_cameras)))

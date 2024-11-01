"""A class that renders a 3D mesh from various cameras."""

import numpy as np
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

from interactive_images.tools import get_cameras_looking_at_points, sph2cart


class MultiviewRenderer:
    """Class that renders a 3D mesh from various cameras."""

    def __init__(
        self,
        image_size: tuple[int, int],
        focal_length: float,
        render_cameras_radious: int,
        render_cameras_number: int = 4,
        render_cameras_elevation_offset: float = 10.0,  # degrees
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = device
        self.render_cameras_number = render_cameras_number
        self.render_cameras_elevation_offset = render_cameras_elevation_offset
        self.dtype = dtype
        self.device = torch.device(device)
        if self.device.type == "cpu":
            self.dtype = torch.float32

        self.render_cameras = self.create_cameras(
            focal_length=focal_length,
            image_size=image_size,
            render_cameras_radious=render_cameras_radious,
            render_cameras_number=render_cameras_number,
            render_cameras_elevation_offset=self.render_cameras_elevation_offset,
        )

        BlendParams.background_color = (1.0, 1.0, 1.0)
        BlendParams.gamma = 0.001

        # Settings for rasterization
        raster_settings = RasterizationSettings(
            image_size=image_size,
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
            location=(torch.cat([render_camera.T for render_camera in self.render_cameras]).tolist()),
        )

        # Setting the mesh rendered
        self.mesh_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.render_cameras,
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=self.render_cameras,
                lights=lights,
            ),
        )

    def create_cameras(
        self,
        focal_length: float,
        image_size: tuple[int, int],
        render_cameras_radious: int,
        render_cameras_number: int = 8,
        render_cameras_elevation_offset: float = 10.0,
    ) -> CamerasBase:
        """Create a set of rendering cameras looking at a 3D mesh from various angles.

        The cameras are distributed evenly around the mesh at a specified elevation offset. Their locations are
        calculated using spherical coordinates, which are then converted to cartesian coordinates.

        Args:
            focal_length (float): The focal length of the cameras.
            image_size (tuple[int, int]): The size (width, height) of the image in pixels.
            render_cameras_radious (int): The spherical coordinates radius of the cameras positions.
            render_cameras_number (int, optional): The number of cameras to create. Defaults to 8.
            render_cameras_elevation_offset (float, optional): The elevation offset of the cameras in degrees.
                Defaults to 10.0.

        Returns:
            CamerasBase: A batch of cameras looking at the 3D mesh.

        """
        return get_cameras_looking_at_points(
            camera_locations=torch.from_numpy(
                # TODO: skip numpy and use torch for the stacking
                np.stack(
                    sph2cart(
                        [360 / render_cameras_number * i for i in range(render_cameras_number)],
                        [90 - render_cameras_elevation_offset] * render_cameras_number,
                        [-render_cameras_radious] * render_cameras_number,
                    ),
                ).T,
            ).to(dtype=self.dtype, device=self.device),
            image_size=torch.tensor(image_size).to(dtype=self.dtype, device=self.device),
            focal_length=focal_length * image_size[0],
        )

    def __call__(
        self,
        mesh: Meshes,
        render_cameras: CamerasBase | None = None,
        mesh_renderer: MeshRenderer | None = None,
    ) -> torch.Tensor:
        """Render the mesh from multiple views.

        Args:
            mesh (Meshes): The mesh to be rendered.
            render_cameras (CamerasBase, optional): The cameras from which the mesh will be rendered. If None, the
                default cameras will be used. Defaults to None.
            mesh_renderer (MeshRenderer, optional): The renderer used for rendering. If None, the default renderer will
                be used. Defaults to None.

        Returns:
            torch.Tensor: The rendered images.

        """
        if render_cameras is None:
            render_cameras = self.render_cameras
        if mesh_renderer is None:
            mesh_renderer = self.mesh_renderer

        return mesh_renderer(join_meshes_as_batch([mesh] * len(render_cameras)))

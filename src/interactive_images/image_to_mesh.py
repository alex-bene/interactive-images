"""A class the reconstructs a 3D mesh from a given 2D image."""

import logging

import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import CamerasBase, TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds
from rich.logging import RichHandler

from interactive_images.models.depth_estimation import DepthEstimationModel
from interactive_images.tools import get_cameras_looking_at_points, save_pointcloud, save_textured_mesh, sph2cart


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Initialize a logger
logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.INFO)


class ImageToMesh:
    """Class that turns a single 2D image into a 3D mesh based on an estimated depth map."""

    def __init__(
        self,
        depth_model: DepthEstimationModel,
        focal_length: float = 0.5,
        render_cameras_number: int = 8,
        render_cameras_elevation_offset: float = 10.0,  # degrees
        edge_threshold: float = 0.01,
    ) -> None:
        self.depth_model = depth_model
        self.device = depth_model.device
        self.dtype = depth_model.dtype
        self.focal_length = focal_length
        self.render_cameras_number = render_cameras_number
        self.render_cameras_elevation_offset = render_cameras_elevation_offset
        self.edge_threshold = edge_threshold

    def generate_faces_from_grid(self, height: int, width: int) -> torch.Tensor:
        """Create mesh triangle indices from a given pixel grid size.

            This function is not and need not be differentiable as triangle indices are
            fixed.
            Source (modified): https://huggingface.co/spaces/shariqfarooq/ZoeDepth/blob/main/gradio_im_to_3d.py

        Args:
        ----
            height: (int) denoting the height of the image.
            width: (int) denoting the width of the image.

        Returns:
        -------
            triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)

        """
        """
        00---01
        |    |
        10---11
        """
        vertex_ids = torch.arange(height * width).reshape(height, width).to(self.device)
        vertex_00 = vertex_ids[: height - 1, : width - 1]
        vertex_01 = vertex_00 + 1
        vertex_10 = vertex_00 + width
        vertex_11 = vertex_00 + width + 1

        return torch.cat(
            torch.stack(
                # counter-clockwise orientation
                # TODO: does the order matter?
                [
                    vertex_00,
                    vertex_10,
                    vertex_01,  # faces_upper_left_triangle
                    vertex_10,
                    vertex_11,
                    vertex_01,  # faces_lower_right_triangle
                ],
            )
            .flatten(1)
            .chunk(2),
            dim=1,
        ).permute(1, 0)

    def edge_threshold_filter(self, vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Only keep faces where all edges are smaller than edge_threshold.

        Will remove stretch artifacts that are caused by inconsistent depth at object borders.

        :param vertices: (N, 3) torch.Tensor of type torch.float32
        :param faces: (M, 3) torch.Tensor of type torch.long
        :param edge_threshold: maximum length per edge (otherwise removes that face).

        :return: filtered faces
        """
        edge_distances = (vertices[faces] - vertices[faces].roll(shifts=-1, dims=1))[:, :, 2]

        mask_small_edge = (edge_distances < self.edge_threshold).all(1)

        return faces[mask_small_edge, :], faces[(edge_distances >= self.edge_threshold).any(1), :]

    @torch.inference_mode
    def predict_depth(self, image: Image.Image) -> torch.Tensor:
        """Predict the depth map for the image."""
        return self.depth_model(image)[0]["predicted_depth"]

    def get_pixel_coordinates_pt3d(self, height: int, width: int) -> torch.Tensor:
        """Get pixel coordinates for pytorch3D.

            For an image with y_resolution and x_resolution, return a tensor of pixel coordinates normalized to lie in
            [0, 1], with the origin (0, 0) in the bottom left corner, the x-axis pointing right, and the y-axis
            pointing up. The top right corner being at (1, 1).

        Returns
        -------
            xy_pix: a meshgrid of values from [0, 1] of shape (y_resolution, x_resolution, 2)

        """
        xs = torch.arange(width - 1, -1, -1)  # Inverted the order for x-coordinates
        ys = torch.arange(height - 1, -1, -1)  # Inverted the order for y-coordinates
        x, y = torch.meshgrid(xs, ys, indexing="xy")

        return torch.cat([x.unsqueeze(dim=2), y.unsqueeze(dim=2)], dim=2).to(self.device)

    def create_cameras(self, depth: torch.Tensor, img_size: tuple[int, int]) -> tuple[CamerasBase, CamerasBase | None]:
        """Create the main camera and additional "render" cameras for later."""
        # TODO: the other cameras should probably not be here but rather on the renderer
        z_center_main = depth[depth.shape[0] // 2, depth.shape[1] // 2]
        z_center_other = (depth.max() + depth.min()) / 2

        return get_cameras_looking_at_points(
            camera_locations=torch.tensor(
                [
                    [0, 0, z_center_main],
                    *torch.tensor(
                        np.stack(
                            sph2cart(
                                [360 / self.render_cameras_number * i for i in range(self.render_cameras_number)],
                                [90 - self.render_cameras_elevation_offset] * self.render_cameras_number,
                                [z_center_other] * self.render_cameras_number,
                            ),
                        ).T,
                    ).tolist(),
                ],
            ).to(dtype=self.dtype, device=self.device),
            image_size=torch.tensor(img_size).to(dtype=self.dtype, device=self.device),
            focal_length=self.focal_length * img_size[0],
        )

    def calculate_world_points(
        self,
        depth: torch.Tensor,
        img_size: tuple[int, int],
        camera: CamerasBase,
    ) -> torch.Tensor:
        """Calculate 3D world points by unprojecting camera-space 2.5D coordinates.

        This function takes the predicted depth map and camera object as input,
        and returns the corresponding 3D world points.

        Args:
        ----
            depth: A 2D tensor representing the predicted depth map.
            img_size: A tuple of integers representing the size of the image.
            camera: A CamerasBase object representing the camera from which the world points are calculated.

        Returns:
        -------
            A 2D tensor representing the 3D world points.

        """
        xy_pix = self.get_pixel_coordinates_pt3d(img_size[1], img_size[0]).flatten(0, -2)
        depth = torch.tensor(depth).unsqueeze(2).flatten(0, -2)
        return camera.unproject_points(torch.cat((xy_pix, depth), dim=1))

    def calculate_faces(
        self,
        world_points: torch.Tensor,
        img_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the triangular faces for the mesh based on the 2D image grid.

        This function generates the triangular faces for the mesh by first creating a grid of faces based
        on the image size. It then filters these faces based on the edge threshold to remove any faces that
        have edges larger than the threshold.

        Args:
        ----
            world_points: A 2D tensor representing the 3D world points of the mesh vertices.
            img_size: A tuple of integers representing the size of the image.

        Returns:
        -------
            A tuple of three tensors:
                - faces_new: The original generated faces.
                - faces_filtered: The faces that have been filtered to have edges smaller than the edge threshold.
                - faces_removed: The faces that have been removed due to having edges larger than the edge threshold.

        """
        faces_new = self.generate_faces_from_grid(img_size[1], img_size[0])
        faces_filtered, faces_removed = self.edge_threshold_filter(world_points, faces_new)
        return faces_new, faces_filtered, faces_removed

    def __call__(
        self,
        image: Image.Image,
        filename: str | None = None,
    ) -> tuple[Meshes, Pointclouds, tuple[CamerasBase, CamerasBase | None]]:
        """Transform a given 2D image to a 3D mesh."""
        # Estimate Depth
        predicted_depth = self.predict_depth(image)

        # Create Cameras
        main_camera, other_cameras = self.create_cameras(predicted_depth, image.size)

        # Calculate 3D World Points
        world_points = self.calculate_world_points(predicted_depth, image.size, main_camera)

        logger.info("Vertices: %s", world_points.shape)

        # Calculate 3D Mesh Faces
        faces_new, faces_filtered, faces_removed = self.calculate_faces(world_points, image.size)

        logger.info("Faces - all: %s", faces_new.shape)
        logger.info("Faces - filtered: %s", faces_filtered.shape)
        logger.info("Faces - removed: %s", faces_removed.shape)

        # Calculate Vertex Colors
        colors = torch.tensor(np.array(image).reshape(-1, 3) / 255).to(torch.float)
        ## "Whitten"-out all vertexes that are part of removed (filtered) faces
        colors[faces_removed.unique()] = torch.full((1, 3), fill_value=255).to(colors)
        textures = TexturesVertex(verts_features=colors.unsqueeze(0))

        # Create Pointcloud and Mesh
        pointcloud = Pointclouds(points=[world_points], features=[colors])
        ## we create the mesh and include the filtered out faces too in order to visualize those regions as white later
        # and know where to use generative filling
        mesh = Meshes(verts=[world_points], faces=[faces_new], textures=textures)
        if filename is not None:
            save_pointcloud(pointcloud, filename)
            save_textured_mesh(mesh, filename)

        return (mesh, pointcloud, (main_camera, other_cameras))

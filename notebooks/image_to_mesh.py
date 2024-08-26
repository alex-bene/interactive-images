"""A class the reconstructs a 3D mesh from a given 2D image"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
import torch
import trimesh
from depth_estimation import DepthEstimationModel, ZoeDepth
from PIL import Image
from pytorch3d.io import IO
from pytorch3d.renderer import (
    CamerasBase,
    FoVPerspectiveCameras,
    HardPhongShader,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    Textures,
    TexturesUV,
    TexturesVertex,
)
from pytorch3d.structures import Meshes, Pointclouds
from tools import get_cameras_looking_at_points, sph2cart

# import open3d as o3d


class ImageToMesh:
    def __init__(
        self,
        depth_model: DepthEstimationModel = ZoeDepth("nyu"),
        focal_length: float = 0.5,
        render_cameras_number: int = 8,
        render_cameras_elevation_offset: float = 10.0,  # degrees
        edge_threshold: float = 0.01,
    ):
        self.depth_model = depth_model
        self.device = depth_model.device
        self.dtype = depth_model.dtype
        self.focal_length = focal_length
        self.render_cameras_number = render_cameras_number
        self.render_cameras_elevation_offset = render_cameras_elevation_offset
        self.edge_threshold = edge_threshold

    def generate_faces_from_grid(self, height: int, width: int):
        # Source (modified): https://huggingface.co/spaces/shariqfarooq/ZoeDepth/blob/main/gradio_im_to_3d.py
        """Creates mesh triangle indices from a given pixel grid size.
            This function is not and need not be differentiable as triangle indices are
            fixed.

        Args:
            h: (int) denoting the height of the image.
            w: (int) denoting the width of the image.

        Returns:
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
                ]
            )
            .flatten(1)
            .chunk(2),
            dim=1,
        ).permute(1, 0)

    def edge_threshold_filter(self, vertices: torch.Tensor, faces: torch.Tensor):
        """
        Only keep faces where all edges are smaller than edge_threshold.
        Will remove stretch artifacts that are caused by inconsistent depth at object borders

        :param vertices: (N, 3) torch.Tensor of type torch.float32
        :param faces: (M, 3) torch.Tensor of type torch.long
        :param edge_threshold: maximum length per edge (otherwise removes that face).

        :return: filtered faces
        """
        edge_distances = (vertices[faces] - vertices[faces].roll(shifts=-1, dims=1))[:, :, 2]

        mask_small_edge = (edge_distances < self.edge_threshold).all(1)

        return faces[mask_small_edge, :], faces[(edge_distances >= self.edge_threshold).any(1), :]

    @torch.inference_mode
    def predict_depth(self, image: Image.Image):
        return self.depth_model(image)[0]["predicted_depth"]

    def get_pixel_coordinates_pt3d(self, height: int, width: int):
        """For an image with y_resolution and x_resolution, return a tensor of pixel coordinates
        normalized to lie in [0, 1], with the origin (0, 0) in the bottom left corner,
        the x-axis pointing right, and the y-axis pointing up. The top right corner
        being at (1, 1).

        Returns:
            xy_pix: a meshgrid of values from [0, 1] of shape
                    (y_resolution, x_resolution, 2)
        """
        xs = torch.arange(width - 1, -1, -1)  # Inverted the order for x-coordinates
        ys = torch.arange(height - 1, -1, -1)  # Inverted the order for y-coordinates
        x, y = torch.meshgrid(xs, ys, indexing="xy")

        return torch.cat([x.unsqueeze(dim=2), y.unsqueeze(dim=2)], dim=2).to(self.device)

    def create_cameras(self, depth: torch.Tensor, img_size: Tuple[int, int]):
        z_center_main = depth[depth.shape[0] // 2, depth.shape[1] // 2]
        z_center_other = (depth.max() + depth.min()) / 2

        return get_cameras_looking_at_points(
            camera_locations=torch.tensor(
                [[0, 0, z_center_main]]
                + torch.tensor(
                    np.stack(
                        sph2cart(
                            [360 / self.render_cameras_number * i for i in range(self.render_cameras_number)],
                            [90 - self.render_cameras_elevation_offset] * self.render_cameras_number,
                            [z_center_other] * self.render_cameras_number,
                        )
                    ).T
                ).tolist()
            ).to(dtype=self.dtype, device=self.device),
            image_size=torch.tensor(img_size).to(dtype=self.dtype, device=self.device),
            focal_length=self.focal_length * img_size[0],
        )

    def calculate_world_points(self, depth: torch.Tensor, img_size: Tuple[int, int], camera: CamerasBase):
        xy_pix = self.get_pixel_coordinates_pt3d(img_size[1], img_size[0]).flatten(0, -2)
        depth = torch.tensor(depth).unsqueeze(2).flatten(0, -2)
        return camera.unproject_points(torch.cat((xy_pix, depth), dim=1))

    def calculate_faces(self, world_points: torch.Tensor, img_size: Tuple[int, int]):
        faces_new = self.generate_faces_from_grid(img_size[1], img_size[0])
        faces_filtered, faces_removed = self.edge_threshold_filter(world_points, faces_new)
        return faces_new, faces_filtered, faces_removed

    def __call__(self, image: Image.Image, log: bool = False, filename: str | None = None):
        # Estimate Depth
        predicted_depth = self.predict_depth(image)

        # Create Cameras
        main_camera, other_cameras = self.create_cameras(predicted_depth, image.size)

        # Calculate 3D World Points
        world_points = self.calculate_world_points(predicted_depth, image.size, main_camera)
        if log:
            print(f"Vertices: {world_points.shape}")

        # Calculate 3D Mesh Faces
        faces_new, faces_filtered, faces_removed = self.calculate_faces(world_points, image.size)
        if log:
            print(f"Faces - all: {faces_new.shape}")
            print(f"Faces - filtered: {faces_filtered.shape}")
            print(f"Faces - removed: {faces_removed.shape}")

        # Calculate Vertex Colors
        colors = torch.tensor(np.array(image).reshape(-1, 3) / 255)
        ## "Whitten"-out all vertexes that are part of removed (filtered) faces
        colors[faces_removed.unique()] = torch.full((1, 3), fill_value=255).to(colors)
        textures = TexturesVertex(verts_features=colors.unsqueeze(0))

        # Create Pointcloud and Mesh
        pointcloud = Pointclouds(points=[world_points], features=[colors])
        ## we create the mesh and include the filtered out faces too in order to visualize those regions as white later
        # and know where to use generative filling
        mesh = Meshes(verts=[world_points], faces=[faces_new], textures=textures)
        if filename is not None:
            filename_mesh = Path(filename).with_suffix(".obj")
            filename_pointcloud = Path(filename).with_suffix(".ply")
            io = IO()
            io.save_mesh(mesh, filename_mesh)
            io.save_pointcloud(pointcloud, filename_pointcloud)

        return (mesh, pointcloud, (main_camera, other_cameras))


if __name__ == "__main__":
    from pytorch3d.vis.plotly_vis import AxisArgs, plot_scene

    # Parameters
    device = "cpu"
    dtype = torch.float32
    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.float16

    focal_length = 0.5
    render_cameras_number = 8
    render_cameras_elevation_offset = 10  # degrees
    edge_threshold = 0.1

    # Load Image
    url = "https://shariqfarooq-zoedepth.hf.space/file=/home/user/app/examples/person_1.jpeg"
    url = "https://shariqfarooq-zoedepth.hf.space/file=/home/user/app/examples/mountains.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)
    image.thumbnail((1024, 1024))

    # Load Model
    model = ZoeDepth("nyu", dtype=dtype, device=device)

    # Load Meshifier
    meshifier = ImageToMesh(
        depth_model=model,
        focal_length=focal_length,
        render_cameras_number=render_cameras_number,
        render_cameras_elevation_offset=render_cameras_elevation_offset,
        edge_threshold=edge_threshold,
    )

    # Image to 3D Mesh
    mesh, point_cloud, (main_camera, other_cameras) = meshifier(image)

    # Visualization
    fig = plot_scene(
        {
            "mesh": {
                "mesh": mesh,
                "main_camera": main_camera,
                "other_cameras": other_cameras,
            },
            "pointcloud": {
                "pointcloud": point_cloud,
                "main_camera": main_camera,
                "other_cameras": other_cameras,
            },
        },
        axis_args=AxisArgs(backgroundcolor="rgb(200,230,200)", showgrid=True, showticklabels=True),
        ncols=1,
        viewpoint_cameras=main_camera,
    )

    fig.update_layout(
        autosize=False,
        width=1200,
        height=1500,
    )

    fig.show()

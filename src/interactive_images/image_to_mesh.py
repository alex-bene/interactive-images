"""A class the reconstructs a 3D mesh from a given 2D image."""

import logging

import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import CamerasBase, TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds
from rich.logging import RichHandler

from interactive_images.models.depth_estimation import DepthEstimationModel
from interactive_images.tools import get_cameras_looking_at_points, save_pointcloud, save_textured_mesh


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
        edge_threshold: float = 0.01,
    ) -> None:
        self.depth_model = depth_model
        self.device = depth_model.device
        self.dtype = depth_model.dtype
        self.edge_threshold = edge_threshold

    @torch.inference_mode
    def predict_depth(self, image: Image.Image) -> torch.Tensor:
        """Predict the depth map for the image."""
        return self.depth_model(image)[0]["predicted_depth"]

    def create_cameras(self, depth: torch.Tensor, image_size: tuple[int, int], focal_length: float) -> CamerasBase:
        """Create the main capture camera.

        This method creates a main capture camera based on the provided depth map, image size, and focal length.
        The camera is positioned at (0, 0, z_center), where z_center is calculated as the depth of the center pixel of
        the image.

        Args:
            depth (torch.Tensor): The depth map of the image. (height x width)
            image_size (tuple[int, int]): The size (width, height) of the image in pixels.
            focal_length (float): The focal length of the cameras.

        Returns:
            CamerasBase: The main capture camera object.

        """
        z_center = depth[depth.shape[0] // 2, depth.shape[1] // 2]

        return get_cameras_looking_at_points(
            camera_locations=torch.zeros(1, 3).to(dtype=self.dtype, device=self.device),
            look_at_points=torch.tensor([[0, 0, z_center]]).to(dtype=self.dtype, device=self.device),
            image_size=torch.tensor(image_size).to(dtype=self.dtype, device=self.device),
            focal_length=focal_length * image_size[0],
        )

    def calculate_world_points(
        self,
        depth: torch.Tensor,
        image_size: tuple[int, int],
        camera: CamerasBase,
    ) -> torch.Tensor:
        """Calculate 3D world points by unprojecting camera-space 2.5D coordinates.

        This function takes the predicted depth map and camera object as input,
        and returns the corresponding 3D world points.

        Args:
            depth (torch.Tensor): Tensor representing the predicted depth map. (height x width)
            image_size (tuple[int, int]): The size (width, height) of the image in pixels.
            camera (CameraBase): The camera from which the world points are calculated.

        Returns:
            torch.Tensor: Tensor representing the 3D world points. (width*height x 3)

        """
        xy_pix = self.get_pixel_coordinates_pt3d(image_size).flatten(0, -2)
        depth = depth.unsqueeze(2).flatten(0, -2)

        return camera.unproject_points(torch.cat((xy_pix, depth), dim=1))

    def get_pixel_coordinates_pt3d(self, image_size: tuple[int, int]) -> torch.Tensor:
        """Get pixel coordinates for pytorch3D.

            For an image with height width, return a tensor of pixel coordinates where the origin (0, 0) is on the
            bottom right corner, the x-axis pointing left and the y-axis pointing up. The top right corner is at
            (height, width).

        Args:
            image_size (tuple[int, int]): The size (width, height) of the image in pixels.

        Returns:
            torch.Tensor: A meshgrid of pixel coordinates. (height x width x 2)

        """
        xs = torch.arange(image_size[0] - 1, -1, -1)  # Inverted the order for x-coordinates
        ys = torch.arange(image_size[1] - 1, -1, -1)  # Inverted the order for y-coordinates
        x, y = torch.meshgrid(xs, ys, indexing="xy")

        return torch.cat([x.unsqueeze(dim=2), y.unsqueeze(dim=2)], dim=2).to(self.device)

    def calculate_faces(
        self,
        world_points: torch.Tensor,
        image_size: tuple[int, int],
        dilate: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the triangular faces for the mesh based on the 2D image grid.

        This function generates the triangular faces for the mesh by first creating a grid of faces based
        on the image size. It then filters these faces based on the edge threshold to remove any faces that
        have edges larger than the threshold. Lastly, to further reduce stretch artifacts, it dilates the faces
        assuming each face to be a node with adjacent faces those that have a common edge.

        Args:
            world_points (torch.Tensor): Tensor representing the 3D world points of the mesh vertices. (vertices x 3)
            image_size (tuple[int, int]): The size (width, height) of the image in pixels.
            dilate (int, optional): Number of dilate steps to perform for the removed faces. Defaults to 1

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the initial generated faces as well as
            the filtered faces and the removed faces based on the edge threshold.

        """
        faces = self.generate_faces_from_grid(image_size)
        faces_to_keep = self.edge_threshold_filter(world_points, faces)
        if dilate == 0:
            return faces, faces[faces_to_keep], faces[~faces_to_keep]

        adjacency_list = self.create_adjacency_list(faces, image_size)
        faces_to_remove = ~faces_to_keep

        # Dilate the removed faces
        for _ in range(dilate):
            dilate_indexes = adjacency_list[faces_to_remove].view(-1).unique()
            dilate_indexes = dilate_indexes[dilate_indexes != -1]
            faces_to_remove[dilate_indexes] = True

        return faces, faces[~faces_to_remove], faces[faces_to_remove]

    def generate_faces_from_grid(self, image_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Create mesh triangle indices from a given pixel grid size.

            This function is not and need not be differentiable as triangle indices are fixed.
            Source (modified): https://huggingface.co/spaces/shariqfarooq/ZoeDepth/blob/main/gradio_im_to_3d.py

            00---01
            |    |
            10---11

        Args:
            image_size (tuple[int, int]): The size (width, height) of the image in pixels.

        Returns:
            torch.Tensor: A tensor representing the faces (indeces of vertices of each triangle). (2*(W-1)*(H-1) x 3)

        """
        width, height = image_size
        vertex_ids = torch.arange(height * width).reshape(height, width).to(self.device)
        vertex_00 = vertex_ids[: height - 1, : width - 1]
        vertex_01 = vertex_00 + 1
        vertex_10 = vertex_00 + width
        vertex_11 = vertex_00 + width + 1

        return torch.cat(
            [
                # counter-clockwise orientation
                torch.stack([vertex_00, vertex_10, vertex_01]).flatten(1),  # faces_upper_left_triangle
                torch.stack([vertex_10, vertex_11, vertex_01]).flatten(1),  # faces_lower_right_triangle
            ],
            dim=1,
        ).permute(1, 0)

    def edge_threshold_filter(self, vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Only keep faces where all edges are smaller than edge_threshold.

        This function will remove stretch artifacts that are caused by inconsistent depth at object borders.

        Args:
            vertices (torch.Tensor): Tensor representing the 3D coordinates of the mesh vertices. (vertices, 3)
            faces (torch.Tensor): Tensor of type `torch.long` representing the triangular faces of the mesh. (faces, 3)
            edge_threshold (float): Maximum length per edge (otherwise removes that face).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the filtered faces and removed faces based on the
            edge threshold.

        """
        edge_distances = (vertices[faces] - vertices[faces].roll(shifts=-1, dims=1))[:, :, 2]
        return (edge_distances < self.edge_threshold).all(1)

    def create_adjacency_list(self, faces: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """Create an adjacency list for a mesh based on the provided faces.

        This function calculates the adjacency list for a mesh by first identifying the upper and lower triangles in
        the mesh. It then calculates the neighbors for each triangle based on their positions in the mesh.

        Args:
            faces (torch.Tensor): Tensor of representing the triangular faces of the mesh. (faces, 3)
            image_size (tuple[int, int]): The size (width, height) of the image in pixels.

        Returns:
            torch.Tensor: A tensor representing the adjacency list for the mesh, where each row corresponds to a face
            in the mesh and each column corresponds to a neighbor of that face.

        """
        # Calculate indices of upper and lower triangles
        upper_triangles = faces[:, 0] < faces[:, 2]
        lower_triangles = ~upper_triangles

        upp = torch.where(upper_triangles)[0].view(image_size[1] - 1, image_size[0] - 1)
        downn = torch.where(lower_triangles)[0].view(image_size[1] - 1, image_size[0] - 1)

        # Upper trianlges can only be neighbors (have common edge) with lower trianlges and vice versa
        # NOTE: If we wanted, we could add as neighbors the faces that only have a single common vertex too
        rightt_of_upp = downn
        leftt_of_upp = downn.roll(1, 1)
        leftt_of_upp[:, 0] = -1
        upp_of_upp = downn.roll(1, 0)
        upp_of_upp[0, :] = -1

        leftt_of_downn = upp
        rightt_of_downn = upp.roll(-1, 1)
        rightt_of_downn[:, -1] = -1
        downn_of_downn = upp.roll(-1, 0)
        downn_of_downn[-1, :] = -1

        upp_neighbors = torch.stack([rightt_of_upp.flatten(), leftt_of_upp.flatten(), upp_of_upp.flatten()], dim=1)
        downn_neighbors = torch.stack(
            [rightt_of_downn.flatten(), leftt_of_downn.flatten(), downn_of_downn.flatten()],
            dim=1,
        )

        return torch.cat([upp_neighbors, downn_neighbors], dim=0)

    def __call__(
        self,
        image: Image.Image,
        filename: str | None = None,
        focal_length: float | None = None,
        dilate: int = 1,
        removed_faces_color: tuple[int, int, int] | list[int] = (256, 256, 256),
    ) -> tuple[Meshes, Pointclouds, CamerasBase]:
        """Transform a given 2D image to a 3D mesh.

        Args:
            image (Image.Image): The input image to be transformed.
            filename (str, optional): The file name to save the results. If None, no results will be saved.
                Defaults to None.
            focal_length (float, optional): The focal length of the camera. If not provided, it will be estimated from
                the image size. Defaults to None.
            dilate (int, optional): Number of dilate steps to perform for the removed faces. Defaults to 1.
            removed_faces_color (tuple[int, int, int] | list[int], optional): Color to use for the removed mesh faces.
                Defaults to [256, 256, 256].

        """
        if focal_length is None:
            msg = "Focal length estimation is not implemented yet."
            raise NotImplementedError(msg)

        # Estimate Depth
        predicted_depth = self.predict_depth(image)

        # Create Cameras
        capture_camera = self.create_cameras(predicted_depth, image.size, focal_length)

        # Calculate 3D World Points
        world_points = self.calculate_world_points(predicted_depth, image.size, capture_camera)

        # Calculate 3D Mesh Faces
        faces_new, faces_filtered, faces_removed = self.calculate_faces(world_points, image.size, dilate=dilate)

        logger.info(
            "Vertices: %s -- Faces (all/keep/remove): %s/%s/%s",
            world_points.shape[0],
            faces_new.shape[0],
            faces_filtered.shape[0],
            faces_removed.shape[0],
        )

        # Calculate Vertex Colors
        colors = torch.tensor(np.array(image).reshape(-1, 3) / 255).to(torch.float)
        ## "Color"-out all vertexes that are part of removed (filtered) faces
        colors[faces_removed.unique()] = torch.tensor([removed_faces_color]).to(colors)
        textures = TexturesVertex(verts_features=colors.unsqueeze(0))

        # Create Pointcloud and Mesh
        pointcloud = Pointclouds(points=[world_points], features=[colors])
        ## we create the mesh and include the filtered out faces too in order to visualize those regions as white later
        # and know where to use generative filling
        mesh = Meshes(verts=[world_points], faces=[faces_new], textures=textures)

        if filename is not None:
            save_pointcloud(pointcloud, filename)
            save_textured_mesh(mesh, filename)

        return mesh, pointcloud, capture_camera

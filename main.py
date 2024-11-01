"""Main script."""

import requests
import torch
from PIL import Image
from pytorch3d.vis.plotly_vis import AxisArgs, plot_scene

from interactive_images import ImageToMesh, MultiviewRenderer
from interactive_images.models import ZoeDepth
from interactive_images.tools import image_grid


if __name__ == "__main__":
    # Parameters
    device = "cpu"
    dtype = torch.float32
    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.float16

    focal_length = 0.5
    render_cameras_number = 2
    render_cameras_elevation_offset = 10  # degrees
    edge_threshold = 0.015
    dilate = 4

    # Load Image
    url = [
        "https://shariqfarooq-zoedepth.hf.space/file=/home/user/app/examples/mountains.jpeg",
        "https://shariqfarooq-zoedepth.hf.space/file=/home/user/app/examples/person_1.jpeg",
    ][0]
    image = Image.open(requests.get(url, stream=True, timeout=5).raw)
    image.thumbnail((1024, 1024))

    # Load Model
    model = ZoeDepth("nyu", dtype=dtype, device=device)

    # Load Meshifier
    meshifier = ImageToMesh(
        depth_model=model,
        edge_threshold=edge_threshold,
    )

    # Image to 3D Mesh
    mesh, point_cloud, main_camera = meshifier(
        image,
        filename=None,
        focal_length=focal_length,
        dilate=dilate,
        removed_faces_color=(0, 125, 50),
    )  # "test")

    offset = torch.tensor([0, 0, mesh.verts_packed().mean(0)[-1]])
    mesh.offset_verts_(-offset)
    point_cloud.offset_(-offset)
    main_camera.T += offset

    renderer = MultiviewRenderer(
        image_size=image.size,
        focal_length=focal_length,
        render_cameras_radious=main_camera.T[0][2].item(),
        render_cameras_number=render_cameras_number,
        render_cameras_elevation_offset=render_cameras_elevation_offset,
        device=device,
        dtype=dtype,
    )

    # Visualization
    fig = plot_scene(
        {
            "mesh": {
                "mesh": mesh,
                "main_camera": main_camera,
                "render_cameras": renderer.render_cameras,
            },
            "pointcloud": {
                "pointcloud": point_cloud,
                "main_camera": main_camera,
                "render_cameras": renderer.render_cameras,
            },
        },
        axis_args=AxisArgs(backgroundcolor="rgb(200,230,200)", showgrid=True, showticklabels=True),
        ncols=1,
        viewpoint_cameras=main_camera,
    )

    fig.update_layout(
        autosize=False,
        width=1000,
        height=2000,
    )

    fig.show()

    images = renderer(mesh)

    images_p = images.clone()
    images_p = (images_p.numpy() * 255).astype("uint8")
    images_p = [Image.fromarray(image_p) for image_p in images_p]
    images_p = [image, *images_p]

    images_grid = image_grid(images_p, 1, render_cameras_number + 1)
    images_grid.save("out.png")
    images_grid.show()

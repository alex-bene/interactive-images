from pathlib import Path

import torch
from PIL import Image

from interactive_images import ImageToMesh
from interactive_images.models import ZoeDepth
from interactive_images.tools import load_pointcloud, load_textured_mesh


def prepare_img():
    return Image.open("./tests/fixtures/tests_samples/zoedepth_mountains.jpeg")


def test_image_to_mesh(tmp_path: Path):
    # Load Image
    image = prepare_img()
    image.thumbnail((1024, 1024))

    # Load Model
    model = ZoeDepth("nyu", dtype=torch.float32, device="cpu")

    # Load Meshifier
    meshifier = ImageToMesh(
        depth_model=model,
        focal_length=0.5,
        render_cameras_number=8,
        render_cameras_elevation_offset=10,
        edge_threshold=0.1,
    )

    # Image to 3D Mesh
    test_filepath = tmp_path / "test.obj"
    mesh, point_cloud, (main_camera, other_cameras) = meshifier(image, filename=test_filepath)

    expected_slices = {
        "verts": torch.tensor(
            [[-3.7404, 2.7998, -1.4861], [-3.7328, 2.7996, -1.4859], [-3.7255, 2.7996, -1.4858]],
        ),
        "faces": torch.tensor([[0, 1024, 1], [1, 1025, 2], [2, 1026, 3]]),
        "features": torch.tensor([[0.6235, 0.6353, 0.5176], [0.5961, 0.6235, 0.4980], [0.6078, 0.6588, 0.5176]]),
    }
    expected_sizes = {
        "verts": torch.Size((785408, 3)),
        "faces": torch.Size((1567236, 3)),
    }

    # Assert Mesh
    assert mesh.verts_list()[0].shape == expected_sizes["verts"]
    assert mesh.faces_list()[0].shape == expected_sizes["faces"]
    assert mesh.textures.verts_features_list()[0].shape == expected_sizes["verts"]
    assert torch.allclose(mesh.verts_list()[0][:3, :], expected_slices["verts"], atol=1e-4)
    assert torch.allclose(mesh.faces_list()[0][:3, :], expected_slices["faces"], atol=1e-4)
    assert torch.allclose(mesh.textures.verts_features_list()[0][:3, :], expected_slices["features"], atol=1e-4)

    # Assert Pointcloud
    assert point_cloud.points_list()[0].shape == expected_sizes["verts"]
    assert point_cloud.features_list()[0].shape == expected_sizes["verts"]
    assert torch.allclose(point_cloud.points_list()[0][:3, :], expected_slices["verts"], atol=1e-4)
    assert torch.allclose(point_cloud.features_list()[0][:3, :], expected_slices["features"], atol=1e-4)

    # Assert Cameras
    assert torch.allclose(main_camera.T, torch.tensor([[-0.0000, -0.0000, 2.2616]]), atol=1e-4)
    assert other_cameras.T.shape == torch.Size([8, 3])
    assert other_cameras.R.shape == torch.Size([8, 3, 3])
    assert torch.allclose(other_cameras.T[1], torch.tensor([-1.6813e-08, 1.2324e-08, 3.2096e00]), atol=1e-4)
    assert torch.allclose(
        other_cameras.R[1],
        torch.tensor([[-0.9923, -0.0152, -0.1228], [0.0000, 0.9924, -0.1228], [0.1237, -0.1218, -0.9848]]),
        atol=1e-4,
    )

    # Assert save/load functions
    loaded_point_cloud = load_pointcloud(test_filepath)
    assert torch.allclose(loaded_point_cloud.points_list()[0], point_cloud.points_list()[0], atol=1e-4)
    assert torch.allclose(loaded_point_cloud.features_list()[0], point_cloud.features_list()[0], atol=1e-4)
    loaded_mesh = load_textured_mesh(test_filepath)
    assert torch.allclose(loaded_mesh.verts_list()[0], mesh.verts_list()[0], atol=1e-4)
    assert torch.allclose(loaded_mesh.faces_list()[0], mesh.faces_list()[0], atol=1e-4)
    assert torch.allclose(
        loaded_mesh.textures.verts_features_list()[0],
        mesh.textures.verts_features_list()[0],
        atol=1e-4,
    )

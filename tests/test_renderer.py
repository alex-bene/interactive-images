import torch

from interactive_images import MultiviewRenderer


def test_image_to_mesh():
    # Load Renderer
    renderer = MultiviewRenderer(
        image_size=(1024, 768),
        focal_length=0.5,
        render_cameras_radious=2,
        render_cameras_number=8,
        render_cameras_elevation_offset=80,
    )

    # Assert Cameras
    assert renderer.render_cameras.T.shape == torch.Size([8, 3])
    assert renderer.render_cameras.R.shape == torch.Size([8, 3, 3])
    assert torch.allclose(renderer.render_cameras.T[2], torch.tensor([0.0, 0.0, 2.0]), atol=1e-4)
    assert torch.allclose(
        renderer.render_cameras.R[2],
        torch.tensor([[1.0000, 0.0000, 0.0000], [0.0000, 0.1736, 0.9848], [0.0000, -0.9848, 0.1736]]),
        atol=1e-4,
    )

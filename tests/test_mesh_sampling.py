from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.scene import sample_mesh_surface_points
from src.scene._internal import mesh_ops


def _write_textured_triangle_pair(root: Path) -> Path:
    texture = np.array([[[255, 0, 0, 255], [0, 255, 0, 255]]], dtype=np.uint8)
    Image.fromarray(texture, mode="RGBA").save(root / "albedo.png")
    (root / "mesh.mtl").write_text("newmtl mesh_material\nmap_Kd albedo.png\n", encoding="utf-8")
    (root / "mesh.obj").write_text(
        "\n".join(
            (
                "mtllib mesh.mtl",
                "usemtl mesh_material",
                "v 0.0 0.0 0.0",
                "v 1.0 0.0 0.0",
                "v 0.0 1.0 0.0",
                "v 2.0 0.0 0.0",
                "v 4.0 0.0 0.0",
                "v 2.0 2.0 0.0",
                "vt 0.0 0.5",
                "vt 0.0 0.5",
                "vt 0.0 0.5",
                "vt 1.0 0.5",
                "vt 1.0 0.5",
                "vt 1.0 0.5",
                "f 1/1 2/2 3/3",
                "f 4/4 5/5 6/6",
                "",
            )
        ),
        encoding="utf-8",
    )
    return root / "mesh.obj"


def test_sample_mesh_surface_points_preserves_textured_colors_in_mesh_space(tmp_path: Path) -> None:
    pytest.importorskip("trimesh")
    mesh_path = _write_textured_triangle_pair(tmp_path)

    positions, colors = sample_mesh_surface_points(mesh_path, 5000, seed=17)

    assert positions.shape == (5000, 3)
    assert positions.dtype == np.float32
    assert colors.shape == (5000, 3)
    assert colors.dtype == np.float32
    assert np.allclose(positions[:, 2], 0.0)
    assert np.all((colors >= 0.0) & (colors <= 1.0))

    unique_colors, counts = np.unique(np.round(colors, decimals=4), axis=0, return_counts=True)
    assert unique_colors.shape == (2, 3)
    assert set(map(tuple, unique_colors.tolist())) == {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)}

    green_fraction = float(counts[np.argmax(unique_colors[:, 1])] / counts.sum())
    assert green_fraction == pytest.approx(0.5, abs=0.04)


def test_sample_mesh_surface_points_samples_triangle_ids_uniformly(tmp_path: Path) -> None:
    pytest.importorskip("trimesh")
    mesh_path = _write_textured_triangle_pair(tmp_path)

    positions, colors = sample_mesh_surface_points(mesh_path, 6000, seed=23)

    assert positions.shape == (6000, 3)
    assert np.allclose(positions[:, 2], 0.0)

    unique_colors, counts = np.unique(np.round(colors, decimals=4), axis=0, return_counts=True)
    assert unique_colors.shape == (2, 3)
    green_fraction = float(counts[np.argmax(unique_colors[:, 1])] / counts.sum())
    assert green_fraction == pytest.approx(0.5, abs=0.04)


def test_sample_mesh_surface_points_reuses_prepared_mesh_cache(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("trimesh")
    mesh_path = _write_textured_triangle_pair(tmp_path)
    trimesh_module = mesh_ops._load_trimesh_module()
    original_load = trimesh_module.load
    load_calls = 0

    def _counting_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return original_load(*args, **kwargs)

    mesh_ops._prepared_mesh_sampling_data.cache_clear()
    monkeypatch.setattr(trimesh_module, "load", _counting_load)

    sample_mesh_surface_points(mesh_path, 128, seed=17)
    sample_mesh_surface_points(mesh_path, 256, seed=18)

    assert load_calls == 1
    mesh_ops._prepared_mesh_sampling_data.cache_clear()


def test_resolve_preferred_mesh_path_uses_matching_binary_rsinfo_sibling(tmp_path: Path) -> None:
    obj_path = tmp_path / "drone.obj"
    glb_path = tmp_path / "mesh.glb"
    obj_path.write_text("o drone\n", encoding="utf-8")
    glb_path.write_bytes(b"glTF")
    (tmp_path / "drone.obj.rsInfo").write_text('<Model componentId="{same-component}" />', encoding="utf-8")
    (tmp_path / "mesh.glb.rsInfo").write_text('<Model componentId="{same-component}" />', encoding="utf-8")

    assert mesh_ops._resolve_preferred_mesh_path(obj_path) == glb_path.resolve()


def test_sample_mesh_surface_points_resolves_preferred_mesh_path_before_loading(tmp_path: Path, monkeypatch) -> None:
    obj_path = tmp_path / "drone.obj"
    glb_path = tmp_path / "mesh.glb"
    obj_path.write_text("o drone\n", encoding="utf-8")
    glb_path.write_bytes(b"glTF")
    (tmp_path / "drone.obj.rsInfo").write_text('<Model componentId="{same-component}" />', encoding="utf-8")
    (tmp_path / "mesh.glb.rsInfo").write_text('<Model componentId="{same-component}" />', encoding="utf-8")

    prepared = mesh_ops._PreparedMeshSamplingData(
        vertices=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        faces=np.array([[0, 0, 0]], dtype=np.int64),
        uv_table=None,
        texture_rgb=None,
        vertex_colors=None,
        face_colors=None,
        material_main_rgb=np.array([0.25, 0.5, 0.75], dtype=np.float32),
    )
    requested_paths: list[str] = []

    monkeypatch.setattr(mesh_ops, "_prepared_mesh_sampling_data", lambda path: requested_paths.append(str(path)) or prepared)

    positions, colors = sample_mesh_surface_points(obj_path, 4, seed=17)

    assert requested_paths == [str(glb_path.resolve())]
    assert positions.shape == (4, 3)
    assert colors.shape == (4, 3)
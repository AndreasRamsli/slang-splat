from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

_DEFAULT_MESH_COLOR = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_PREFERRED_BINARY_MESH_SUFFIXES = (".glb", ".gltf")


@dataclass(frozen=True, slots=True)
class _PreparedMeshSamplingData:
    vertices: np.ndarray
    faces: np.ndarray
    uv_table: np.ndarray | None
    texture_rgb: np.ndarray | None
    vertex_colors: np.ndarray | None
    face_colors: np.ndarray | None
    material_main_rgb: np.ndarray | None


def _load_trimesh_module():
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Mesh initialization requires the trimesh package.") from exc
    return trimesh


def _rsinfo_path(mesh_path: Path) -> Path:
    return Path(f"{Path(mesh_path).resolve()}.rsInfo")


def _rsinfo_component_id(rsinfo_path: Path) -> str | None:
    path = Path(rsinfo_path).resolve()
    if not path.exists() or not path.is_file():
        return None
    try:
        document = path.read_text(encoding="utf-8")
    except (ET.ParseError, OSError, UnicodeDecodeError):
        return None
    try:
        root = ET.fromstring(f"<RealityScanInfo>{document}</RealityScanInfo>")
    except ET.ParseError:
        return None
    model = root.find("Model")
    if model is None:
        return None
    value = str(model.attrib.get("componentId", "")).strip()
    return value or None


def _resolve_preferred_mesh_path(mesh_path: Path) -> Path:
    resolved = Path(mesh_path).resolve()
    if resolved.suffix.lower() in _PREFERRED_BINARY_MESH_SUFFIXES:
        return resolved

    for suffix in _PREFERRED_BINARY_MESH_SUFFIXES:
        sibling = resolved.with_suffix(suffix)
        if sibling.exists() and sibling.is_file():
            return sibling.resolve()

    source_component_id = _rsinfo_component_id(_rsinfo_path(resolved))
    if source_component_id is None:
        return resolved

    candidates: list[Path] = []
    for suffix in _PREFERRED_BINARY_MESH_SUFFIXES:
        candidates.extend(sorted(resolved.parent.glob(f"*{suffix}.rsInfo")))
    for candidate_rsinfo in candidates:
        candidate_asset = candidate_rsinfo.with_suffix("").resolve()
        if candidate_asset == resolved or not candidate_asset.exists() or not candidate_asset.is_file():
            continue
        if _rsinfo_component_id(candidate_rsinfo) == source_component_id:
            return candidate_asset
    return resolved


def _normalize_rgb(colors: np.ndarray) -> np.ndarray:
    rgb = np.asarray(colors, dtype=np.float32)
    if rgb.ndim == 1:
        rgb = rgb[None, :]
    if rgb.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if rgb.shape[1] < 3:
        raise RuntimeError("Mesh color payload must provide at least RGB channels.")
    scale = 255.0 if float(np.max(rgb[:, :3])) > 1.0 else 1.0
    return np.ascontiguousarray(np.clip(rgb[:, :3] / scale, 0.0, 1.0), dtype=np.float32)


def _material_image(material: object) -> object | None:
    if material is None:
        return None
    image = getattr(material, "image", None)
    if image is not None:
        return image
    return getattr(material, "baseColorTexture", None)


def _material_main_rgb(material: object) -> np.ndarray | None:
    if material is None:
        return None
    main_color = getattr(material, "main_color", None)
    if main_color is None:
        return None
    return _normalize_rgb(np.asarray(main_color, dtype=np.float32).reshape(1, -1))[0]


def _as_single_mesh(mesh_source: object, trimesh_module) -> object:
    if isinstance(mesh_source, trimesh_module.Trimesh):
        return mesh_source.copy()
    if isinstance(mesh_source, trimesh_module.Scene):
        dumped = mesh_source.dump(concatenate=False)
        meshes = dumped if isinstance(dumped, list) else [dumped]
        valid_meshes = [mesh for mesh in meshes if isinstance(mesh, trimesh_module.Trimesh) and np.asarray(getattr(mesh, "faces", ())).size > 0]
        if len(valid_meshes) == 0:
            raise RuntimeError("Mesh initialization requires at least one triangular mesh geometry.")
        return valid_meshes[0].copy() if len(valid_meshes) == 1 else trimesh_module.util.concatenate(valid_meshes)
    raise RuntimeError("Mesh initialization requires a triangle mesh file.")


def _sample_barycentrics(count: int, rng: np.random.Generator) -> np.ndarray:
    random_uv = rng.random((count, 2), dtype=np.float32)
    sqrt_u = np.sqrt(random_uv[:, :1])
    return np.ascontiguousarray(
        np.concatenate((1.0 - sqrt_u, sqrt_u * (1.0 - random_uv[:, 1:2]), sqrt_u * random_uv[:, 1:2]), axis=1),
        dtype=np.float32,
    )


def _sample_texture_rgb(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    if texture.ndim != 3 or texture.shape[2] < 3:
        raise RuntimeError("Mesh texture payload must provide RGB channels.")
    height, width = texture.shape[:2]
    x = np.asarray(uv[:, 0], dtype=np.float32) * np.float32(width - 1)
    y = (np.float32(1.0) - np.asarray(uv[:, 1], dtype=np.float32)) * np.float32(height - 1)

    x_floor = np.floor(x).astype(np.int64) % width
    y_floor = np.floor(y).astype(np.int64) % height
    x_ceil = np.ceil(x).astype(np.int64) % width
    y_ceil = np.ceil(y).astype(np.int64) % height

    dx = np.mod(x, np.float32(width)) - x_floor.astype(np.float32)
    dy = np.mod(y, np.float32(height)) - y_floor.astype(np.float32)

    colors00 = texture[y_floor, x_floor, :3]
    colors01 = texture[y_floor, x_ceil, :3]
    colors10 = texture[y_ceil, x_floor, :3]
    colors11 = texture[y_ceil, x_ceil, :3]

    weight00 = ((1.0 - dx) * (1.0 - dy))[:, None]
    weight01 = (dx * (1.0 - dy))[:, None]
    weight10 = ((1.0 - dx) * dy)[:, None]
    weight11 = (dx * dy)[:, None]

    return np.ascontiguousarray((weight00 * colors00 + weight01 * colors01 + weight10 * colors10 + weight11 * colors11) / 255.0, dtype=np.float32)


def _normalized_color_table(colors: object, expected_count: int | None = None) -> np.ndarray | None:
    if colors is None:
        return None
    color_table = np.asarray(colors, dtype=np.float32)
    if color_table.ndim != 2 or color_table.shape[1] < 3:
        return None
    if expected_count is not None and color_table.shape[0] != expected_count:
        return None
    return _normalize_rgb(color_table)


@lru_cache(maxsize=4)
def _prepared_mesh_sampling_data(mesh_path: str) -> _PreparedMeshSamplingData:
    trimesh_module = _load_trimesh_module()
    mesh = _as_single_mesh(trimesh_module.load(Path(mesh_path), force="scene", process=False), trimesh_module)
    faces = np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.int64))
    vertices = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float32))
    if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] != 3:
        raise RuntimeError("Mesh initialization requires a triangular mesh.")

    visual = getattr(mesh, "visual", None)
    uv = None if visual is None else getattr(visual, "uv", None)
    material = None if visual is None else getattr(visual, "material", None)
    image = _material_image(material)
    uv_table = np.asarray(uv, dtype=np.float32)[:, :2] if uv is not None else None
    if uv_table is not None and (uv_table.ndim != 2 or uv_table.shape[0] != vertices.shape[0] or uv_table.shape[1] != 2):
        uv_table = None
    texture_rgb = None if image is None else np.ascontiguousarray(np.asarray(image.convert("RGBA"), dtype=np.float32)[:, :, :3])
    vertex_colors = _normalized_color_table(None if visual is None else getattr(visual, "vertex_colors", None), vertices.shape[0])
    face_colors = _normalized_color_table(None if visual is None else getattr(visual, "face_colors", None), faces.shape[0])
    material_main_rgb = _material_main_rgb(material)
    return _PreparedMeshSamplingData(
        vertices=vertices,
        faces=faces,
        uv_table=None if uv_table is None else np.ascontiguousarray(uv_table, dtype=np.float32),
        texture_rgb=texture_rgb,
        vertex_colors=vertex_colors,
        face_colors=face_colors,
        material_main_rgb=material_main_rgb,
    )


def _sample_texture_colors(data: _PreparedMeshSamplingData, sampled_faces: np.ndarray, barycentrics: np.ndarray) -> np.ndarray | None:
    if data.uv_table is None or data.texture_rgb is None:
        return None
    sampled_uv = np.sum(data.uv_table[sampled_faces] * barycentrics[:, :, None], axis=1)
    return _sample_texture_rgb(data.texture_rgb, sampled_uv)


def _sample_vertex_colors(data: _PreparedMeshSamplingData, sampled_faces: np.ndarray, barycentrics: np.ndarray) -> np.ndarray | None:
    if data.vertex_colors is None:
        return None
    return np.ascontiguousarray(np.sum(data.vertex_colors[sampled_faces] * barycentrics[:, :, None], axis=1), dtype=np.float32)


def _sample_face_colors(data: _PreparedMeshSamplingData, sampled_face_indices: np.ndarray) -> np.ndarray | None:
    if data.face_colors is None:
        return None
    return np.ascontiguousarray(data.face_colors[sampled_face_indices], dtype=np.float32)


def _sample_mesh_colors(data: _PreparedMeshSamplingData, sampled_faces: np.ndarray, sampled_face_indices: np.ndarray, barycentrics: np.ndarray) -> np.ndarray:
    textured = _sample_texture_colors(data, sampled_faces, barycentrics)
    if textured is not None:
        return textured
    vertex = _sample_vertex_colors(data, sampled_faces, barycentrics)
    if vertex is not None:
        return vertex
    face = _sample_face_colors(data, sampled_face_indices)
    if face is not None:
        return face
    main_rgb = data.material_main_rgb
    fallback = _DEFAULT_MESH_COLOR if main_rgb is None else main_rgb.astype(np.float32, copy=False)
    return np.ascontiguousarray(np.repeat(fallback[None, :], sampled_faces.shape[0], axis=0), dtype=np.float32)


def sample_mesh_surface_points(mesh_path: Path, point_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(point_count)
    if count <= 0:
        raise RuntimeError("Mesh initialization requires a positive point count.")
    data = _prepared_mesh_sampling_data(str(_resolve_preferred_mesh_path(Path(mesh_path))))
    faces = data.faces
    vertices = data.vertices
    if faces.shape[0] == 0:
        raise RuntimeError("Mesh initialization requires at least one triangle.")
    rng = np.random.default_rng(int(seed))
    sampled_face_indices = rng.integers(0, faces.shape[0], size=count, dtype=np.int64)
    sampled_faces = faces[sampled_face_indices]
    barycentrics = _sample_barycentrics(count, rng)
    positions = np.ascontiguousarray(np.sum(vertices[sampled_faces] * barycentrics[:, :, None], axis=1), dtype=np.float32)
    colors = _sample_mesh_colors(data, sampled_faces, sampled_face_indices, barycentrics)
    return positions, np.ascontiguousarray(colors, dtype=np.float32)
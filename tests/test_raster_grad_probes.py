from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy

from src.renderer import GaussianRenderer
from src.utility import SHADER_ROOT, buffer_to_numpy, load_compute_kernels
from tests.test_training_kernels import _log_sigma, _make_frame, _make_scene

_BUFFER_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
_RASTER_COMPONENT_COUNT = 13
_GAUSSIAN_GRAD_COMPONENT_COUNT = 11
_BACKGROUND = np.zeros((3,), dtype=np.float32)
_SCREEN_POS = np.array([41.5, 24.5], dtype=np.float32)
_DIRECT_ALPHA_GRAD = np.array([[1.0, 0.0]], dtype=np.float32)
_DIRECT_DEPTH_GRAD = np.array([[0.0, 1.0]], dtype=np.float32)
_DIRECT_COMBINED_GRAD = np.array([[1.0, 0.5]], dtype=np.float32)


def _probe_kernels(device: spy.Device) -> dict[str, spy.ComputeKernel]:
    return load_compute_kernels(
        device,
        SHADER_ROOT / "utility" / "raster_grad_probe_stage.slang",
        {
            "eval": "csEvalRasterAlphaDepth",
            "backprop_eval": "csBackpropRasterEvalAlphaDepth",
            "build_forward": "csBuildRasterGaussianForward",
            "backprop_build": "csBackpropBuildRasterGaussian",
            "flush_float": "csFlushCachedRasterGradFloatFromInput",
            "flush_fixed": "csFlushCachedRasterGradFixedFromInput",
            "load_float": "csLoadCachedRasterGradFloatToOutput",
            "load_fixed": "csLoadCachedRasterGradFixedToOutput",
        },
    )


def _create_buffer(device: spy.Device, values: np.ndarray, dtype: np.dtype = np.float32) -> spy.Buffer:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    buffer = device.create_buffer(size=max(int(array.nbytes), int(np.dtype(dtype).itemsize)), usage=_BUFFER_USAGE)
    buffer.copy_from_numpy(array.reshape(-1))
    return buffer


def _read_matrix(buffer: spy.Buffer, rows: int, cols: int, dtype: np.dtype = np.float32) -> np.ndarray:
    flat = np.asarray(buffer_to_numpy(buffer, dtype), dtype=dtype)
    return flat[: max(int(rows), 1) * int(cols)].reshape(max(int(rows), 1), int(cols))[: int(rows)].copy()


def _rotation_y_90() -> np.ndarray:
    return np.array([math.cos(math.pi / 4.0), 0.0, math.sin(math.pi / 4.0), 0.0], dtype=np.float32)


def _make_probe_renderer(device: spy.Device, tmp_path: Path, *, rotation_wxyz: np.ndarray, scales: np.ndarray, image_name: str) -> tuple[GaussianRenderer, object]:
    scene = _make_scene(count=1, seed=111)
    scene.positions[:] = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    scene.scales[:] = _log_sigma(np.asarray(scales, dtype=np.float32).reshape(1, 3))
    scene.rotations[:] = np.asarray(rotation_wxyz, dtype=np.float32).reshape(1, 4)
    scene.opacities[:] = np.array([2.0], dtype=np.float32)
    scene.colors[:] = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    frame = _make_frame(tmp_path, width=64, height=64, image_name=image_name, image_id=0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        list_capacity_multiplier=16,
        alpha_cutoff=0.0,
        cached_raster_grad_include_depth=True,
    )
    renderer.set_scene(scene)
    return renderer, frame.make_camera(near=0.1, far=20.0)


def _camera_ray_camera_space(camera: object, width: int, height: int) -> np.ndarray:
    ray_world = camera.screen_to_world_ray(_SCREEN_POS, width, height)
    return np.asarray(camera.world_to_camera(ray_world), dtype=np.float32).reshape(1, 3)


def _build_raster_forward(renderer: GaussianRenderer, camera: object, kernels: dict[str, spy.ComputeKernel]) -> np.ndarray:
    out = _create_buffer(renderer.device, np.zeros((_RASTER_COMPONENT_COUNT,), dtype=np.float32))
    enc = renderer.device.create_command_encoder()
    kernels["build_forward"].dispatch(
        thread_count=spy.uint3(1, 1, 1),
        vars={
            **renderer._scene_vars(),
            **renderer._prepass_uniforms(1),
            **renderer._raster_uniforms(_BACKGROUND),
            **renderer._anisotropy_uniforms(),
            **renderer._camera_uniforms(camera),
            "g_OutputRasterGaussian": out,
        },
        command_encoder=enc,
    )
    renderer.device.submit_command_buffer(enc.finish())
    renderer.device.wait()
    return _read_matrix(out, 1, _RASTER_COMPONENT_COUNT)


def _eval_raster_alpha_depth(renderer: GaussianRenderer, kernels: dict[str, spy.ComputeKernel], raster: np.ndarray, ray_dirs: np.ndarray) -> np.ndarray:
    raster_buffer = _create_buffer(renderer.device, raster, np.float32)
    ray_buffer = _create_buffer(renderer.device, ray_dirs, np.float32)
    out = _create_buffer(renderer.device, np.zeros((2,), dtype=np.float32))
    enc = renderer.device.create_command_encoder()
    kernels["eval"].dispatch(
        thread_count=spy.uint3(1, 1, 1),
        vars={
            "g_InputRasterGaussian": raster_buffer,
            "g_InputRayDirs": ray_buffer,
            "g_OutputEval": out,
            **renderer._prepass_uniforms(1),
            **renderer._raster_uniforms(_BACKGROUND),
        },
        command_encoder=enc,
    )
    renderer.device.submit_command_buffer(enc.finish())
    renderer.device.wait()
    return _read_matrix(out, 1, 2)


def _backprop_raster_eval(renderer: GaussianRenderer, kernels: dict[str, spy.ComputeKernel], raster: np.ndarray, ray_dirs: np.ndarray, eval_grad: np.ndarray) -> np.ndarray:
    raster_buffer = _create_buffer(renderer.device, raster, np.float32)
    ray_buffer = _create_buffer(renderer.device, ray_dirs, np.float32)
    grad_buffer = _create_buffer(renderer.device, eval_grad, np.float32)
    out = _create_buffer(renderer.device, np.zeros((_RASTER_COMPONENT_COUNT,), dtype=np.float32))
    enc = renderer.device.create_command_encoder()
    kernels["backprop_eval"].dispatch(
        thread_count=spy.uint3(1, 1, 1),
        vars={
            "g_InputRasterGaussian": raster_buffer,
            "g_InputRayDirs": ray_buffer,
            "g_InputEvalGrad": grad_buffer,
            "g_OutputRasterGaussian": out,
            **renderer._prepass_uniforms(1),
            **renderer._raster_uniforms(_BACKGROUND),
        },
        command_encoder=enc,
    )
    renderer.device.submit_command_buffer(enc.finish())
    renderer.device.wait()
    return _read_matrix(out, 1, _RASTER_COMPONENT_COUNT)


def _backprop_build_raster(renderer: GaussianRenderer, camera: object, kernels: dict[str, spy.ComputeKernel], raster_grad: np.ndarray) -> np.ndarray:
    grad_buffer = _create_buffer(renderer.device, raster_grad, np.float32)
    out = _create_buffer(renderer.device, np.zeros((_GAUSSIAN_GRAD_COMPONENT_COUNT,), dtype=np.float32))
    enc = renderer.device.create_command_encoder()
    kernels["backprop_build"].dispatch(
        thread_count=spy.uint3(1, 1, 1),
        vars={
            **renderer._scene_vars(),
            **renderer._prepass_uniforms(1),
            **renderer._raster_uniforms(_BACKGROUND),
            **renderer._anisotropy_uniforms(),
            **renderer._camera_uniforms(camera),
            "g_InputRasterGrad": grad_buffer,
            "g_OutputGaussianGrad": out,
        },
        command_encoder=enc,
    )
    renderer.device.submit_command_buffer(enc.finish())
    renderer.device.wait()
    return _read_matrix(out, 1, _GAUSSIAN_GRAD_COMPONENT_COUNT)


def _flush_and_load_cached_raster(renderer: GaussianRenderer, kernels: dict[str, spy.ComputeKernel], raster: np.ndarray, raster_grad: np.ndarray, mode: str) -> np.ndarray:
    renderer.cached_raster_grad_atomic_mode = mode
    raster_grad_buffer = _create_buffer(renderer.device, raster_grad, np.float32)
    output = _create_buffer(renderer.device, np.zeros((_RASTER_COMPONENT_COUNT,), dtype=np.float32))
    if mode == renderer.CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED:
        renderer.work_buffers["cached_raster_grads_fixed"].copy_from_numpy(np.zeros((_RASTER_COMPONENT_COUNT,), dtype=np.int32))
        renderer.work_buffers["raster_cache"].copy_from_numpy(np.asarray(raster.reshape(-1), dtype=np.float32))
        flush_kernel = kernels["flush_fixed"]
        load_kernel = kernels["load_fixed"]
    else:
        renderer.work_buffers["cached_raster_grads_float"].copy_from_numpy(np.zeros((_RASTER_COMPONENT_COUNT,), dtype=np.float32))
        flush_kernel = kernels["flush_float"]
        load_kernel = kernels["load_float"]
    enc = renderer.device.create_command_encoder()
    flush_kernel.dispatch(
        thread_count=spy.uint3(1, 1, 1),
        vars={
            **renderer._prepass_uniforms(1),
            **renderer._raster_uniforms(_BACKGROUND),
            **renderer._raster_cache_vars(),
            **renderer._raster_grad_vars(),
            **renderer._raster_grad_fixed_range_vars(),
            "g_GradientStats": renderer.work_buffers["debug_grad_stats"],
            "g_InputRasterGrad": raster_grad_buffer,
        },
        command_encoder=enc,
    )
    load_kernel.dispatch(
        thread_count=spy.uint3(1, 1, 1),
        vars={
            **renderer._prepass_uniforms(1),
            **renderer._raster_cache_vars(),
            **renderer._raster_grad_vars(),
            **renderer._raster_grad_fixed_range_vars(),
            "g_OutputRasterGaussian": output,
        },
        command_encoder=enc,
    )
    renderer.device.submit_command_buffer(enc.finish())
    renderer.device.wait()
    return _read_matrix(output, 1, _RASTER_COMPONENT_COUNT)


def _backprop_cached_stage(renderer: GaussianRenderer, camera: object) -> np.ndarray:
    renderer.work_buffers["param_grads"].copy_from_numpy(np.zeros((renderer.packed_trainable_param_count,), dtype=np.float32))
    enc = renderer.device.create_command_encoder()
    renderer._raster_grad_shader_set().backprop.dispatch(
        thread_count=spy.uint3(1, 1, 1),
        vars={
            **renderer._scene_vars(),
            **renderer._raster_cache_vars(),
            **renderer._raster_grad_vars(),
            **renderer._raster_grad_decode_scale_var(1.0),
            **renderer._raster_grad_fixed_range_vars(),
            **renderer._prepass_uniforms(1),
            **renderer._raster_uniforms(_BACKGROUND),
            **renderer._anisotropy_uniforms(),
            **renderer._camera_uniforms(camera),
        },
        command_encoder=enc,
    )
    renderer.device.submit_command_buffer(enc.finish())
    renderer.device.wait()
    grads = renderer.read_grad_groups(1)
    return np.concatenate(
        (
            np.asarray(grads["grad_positions"][0, :3], dtype=np.float32),
            np.asarray(grads["grad_scales"][0, :3], dtype=np.float32),
            np.asarray(grads["grad_rotations"][0, :4], dtype=np.float32),
            np.asarray([grads["grad_color_alpha"][0, 3]], dtype=np.float32),
        )
    )


def test_direct_raster_eval_backprop_uses_actual_camera_ray(device, tmp_path: Path) -> None:
    kernels = _probe_kernels(device)
    renderer, camera = _make_probe_renderer(
        device,
        tmp_path,
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        scales=np.array([0.03, 0.03, 0.4], dtype=np.float32),
        image_name="raster_grad_probe_eval.png",
    )
    try:
        ray_dirs = _camera_ray_camera_space(camera, renderer.width, renderer.height)
        raster = _build_raster_forward(renderer, camera, kernels)
        eval_value = _eval_raster_alpha_depth(renderer, kernels, raster, ray_dirs)
        alpha_grad = _backprop_raster_eval(renderer, kernels, raster, ray_dirs, _DIRECT_ALPHA_GRAD)
        depth_grad = _backprop_raster_eval(renderer, kernels, raster, ray_dirs, _DIRECT_DEPTH_GRAD)

        assert float(eval_value[0, 0]) > 0.0
        assert float(eval_value[0, 1]) > 0.0
        assert np.all(np.isfinite(alpha_grad))
        assert np.all(np.isfinite(depth_grad))
        assert np.any(np.abs(alpha_grad[0, :9]) > 0.0)
        assert np.any(np.abs(depth_grad[0, :9]) > 0.0)
    finally:
        renderer.clear_scene_resources()


def test_cached_float_backprop_matches_direct_build_backprop_for_actual_ray_gradient(device, tmp_path: Path) -> None:
    kernels = _probe_kernels(device)
    renderer, camera = _make_probe_renderer(
        device,
        tmp_path,
        rotation_wxyz=_rotation_y_90(),
        scales=np.array([0.03, 0.03, 0.4], dtype=np.float32),
        image_name="raster_grad_probe_float.png",
    )
    try:
        try:
            renderer.cached_raster_grad_atomic_mode = renderer.CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT
        except RuntimeError as exc:
            pytest.skip(str(exc))
        ray_dirs = _camera_ray_camera_space(camera, renderer.width, renderer.height)
        raster = _build_raster_forward(renderer, camera, kernels)
        raster_grad = _backprop_raster_eval(renderer, kernels, raster, ray_dirs, _DIRECT_COMBINED_GRAD)
        direct = _backprop_build_raster(renderer, camera, kernels, raster_grad)[0]
        cached = _flush_and_load_cached_raster(renderer, kernels, raster, raster_grad, renderer.CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT)[0]
        staged = _backprop_cached_stage(renderer, camera)

        np.testing.assert_allclose(cached, raster_grad[0], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(staged[:3], direct[:3], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(staged[3:6], direct[3:6], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(staged[6:10], direct[6:10], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(staged[10], direct[10], rtol=0.0, atol=1e-6)
    finally:
        renderer.clear_scene_resources()


def test_cached_fixed_backprop_tracks_direct_build_backprop_for_actual_ray_gradient(device, tmp_path: Path) -> None:
    kernels = _probe_kernels(device)
    renderer, camera = _make_probe_renderer(
        device,
        tmp_path,
        rotation_wxyz=_rotation_y_90(),
        scales=np.array([0.03, 0.03, 0.4], dtype=np.float32),
        image_name="raster_grad_probe_fixed.png",
    )
    try:
        renderer.cached_raster_grad_atomic_mode = renderer.CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED
        ray_dirs = _camera_ray_camera_space(camera, renderer.width, renderer.height)
        raster = _build_raster_forward(renderer, camera, kernels)
        raster_grad = _backprop_raster_eval(renderer, kernels, raster, ray_dirs, _DIRECT_COMBINED_GRAD)
        direct = _backprop_build_raster(renderer, camera, kernels, raster_grad)[0]
        cached = _flush_and_load_cached_raster(renderer, kernels, raster, raster_grad, renderer.CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED)[0]
        staged = _backprop_cached_stage(renderer, camera)

        np.testing.assert_allclose(cached, raster_grad[0], rtol=0.0, atol=3e-3)
        np.testing.assert_allclose(staged[:3], direct[:3], rtol=0.0, atol=2e-3)
        np.testing.assert_allclose(staged[3:6], direct[3:6], rtol=0.0, atol=2e-3)
        np.testing.assert_allclose(staged[10], direct[10], rtol=0.0, atol=2e-3)
        assert np.all(np.isfinite(staged[6:10]))
        assert float(np.max(np.abs(staged[6:10]))) > 0.0
    finally:
        renderer.clear_scene_resources()
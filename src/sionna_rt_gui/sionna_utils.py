#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os

import drjit as dr
import mitsuba as mi
import numpy as np
import polyscope as ps
from sionna import rt
from sionna.rt.constants import DEFAULT_TRANSMITTER_COLOR, DEFAULT_RECEIVER_COLOR
from sionna.rt.utils.geometry import rotation_matrix
from sionna.rt.utils.render import scene_scale

from . import SCENES_DIR
from .config import RadioMapConfig, DEFAULT_SLICE_PLANE_NAME
from .ps_utils import supports_direct_update_from_device
from .rm_utils import radio_map_texture


ITU_TO_PS_MATERIAL = {
    "marble": "ceramic",
    "concrete": "clay",
    "wood": "clay",
    "metal": "candy",
    "brick": "clay",
    "glass": "ceramic",
    "floorboard": "clay",
    "ceiling_board": "clay",
    "chipboard": "clay",
    "plasterboard": "clay",
    "plywood": "clay",
    "very_dry_ground": "clay",
    "medium_dry_ground": "clay",
    "wet_ground": "clay",
}

# Thermal noise power helper (kTB)
K_BOLTZMANN = 1.380649e-23  # J/K


def get_built_in_scenes() -> dict[str, str]:
    result = {}
    for var_name in dir(rt.scene):
        var = getattr(rt.scene, var_name)
        if isinstance(var, str) and var.endswith(".xml"):
            result[var_name] = var

    for dir_name in os.listdir(SCENES_DIR):
        scene_fname = os.path.join(SCENES_DIR, dir_name, f"{dir_name}.xml")
        if os.path.exists(scene_fname):
            # Note: this may override a built-in Sionna RT scene.
            result[dir_name] = scene_fname

    return result


def add_scene_to_polyscope(scene: rt.Scene, ps_groups: dict[str, ps.Group]):
    # Add the meshes to Polyscope
    # TODO: apply consistent materials (based on radio material)
    for mesh in scene.mi_scene.shapes():
        mat = mesh.bsdf()
        ps_mat = None
        if isinstance(mat, rt.RadioMaterialBase):
            color = mat.color
            # TODO: use fancier materials for rasterization
            # if isinstance(mat, rt.ITURadioMaterial):
            #     ps_mat = ITU_TO_PS_MATERIAL.get(mat.itu_type)
        else:
            color = (0.65, 0.65, 0.65)

        vertices = mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
        faces = mesh.faces_buffer().numpy().reshape(-1, 3)
        struct = ps.register_surface_mesh(
            mesh.id(), vertices, faces, color=color, material=ps_mat
        )
        struct.add_to_group(ps_groups["scene"])
        struct.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME, False)


def set_or_update_radio_devices_polyscope(
    radio_devices: dict[str, rt.RadioDevice],
    is_transmitter: bool,
    gui: "SionnaRtGui",
):
    name = "Transmitters" if is_transmitter else "Receivers"
    if not radio_devices:
        if ps.has_point_cloud(name):
            ps.get_point_cloud(name).remove()
        return

    position_np = np.concatenate(
        [rd.position.numpy().T for rd in radio_devices.values()], axis=0
    )
    struct = None
    if ps.has_point_cloud(name):
        # Update existing point cloud (only possible if it has the same size)
        candidate = ps.get_point_cloud(name)
        if candidate.n_points() == position_np.shape[0]:
            candidate.update_point_positions(position_np)
            struct = candidate

    if struct is None:
        display_radius = max(0.001 * scene_scale(gui.scene), 1)
        struct = ps.register_point_cloud(
            name,
            position_np,
            color=(
                DEFAULT_TRANSMITTER_COLOR if is_transmitter else DEFAULT_RECEIVER_COLOR
            ),
        )
        struct.set_radius(display_radius, relative=False)
        struct.add_to_group(gui.ps_groups["rd"])
        struct.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME, True)

    # Update orientations
    rd_orientations = np.array(
        [
            # TODO: maybe use this when the new version of Mitsuba is released
            # dr.quat_apply(
            #     dr.euler_to_quat(rd.orientation.numpy().T[0]),
            #     mi.ScalarVector3f(0, 0, 1),
            # )
            rotation_matrix(rd.orientation).numpy()[:, 0].T[0]
            for rd in radio_devices.values()
        ]
    )
    # Don't show orientation if it's the default value (all zero Euler angles)
    is_default = np.all(
        np.array([rd.orientation.numpy()[0] for rd in radio_devices.values()]) == 0,
        axis=1,
    )
    rd_orientations[is_default, :] = 0

    sphere_radius = struct.get_radius()
    struct.add_vector_quantity(
        name + "_orientation",
        rd_orientations,
        color=(0.6, 0.6, 0.6),
        enabled=True,
        # Note: these are relative to the Polyscope scene scale
        radius=0.3 * sphere_radius / ps.get_length_scale(),
        length=2.5 * sphere_radius / ps.get_length_scale(),
    )

    # Also update per-point colors
    rd_colors = np.array([rd.color for rd in radio_devices.values()])
    struct.add_color_quantity(
        name + "_colors",
        rd_colors,
        enabled=True,
    )


def add_radio_map_to_polyscope(
    name: str,
    radio_map: rt.RadioMap | None,
    ps_groups: dict[str, ps.Group],
    cfg: RadioMapConfig,
    direct_update_from_device: bool = True,
    use_alpha: bool = True,
):
    if radio_map is None:
        return

    direct_update_from_device &= supports_direct_update_from_device()

    if isinstance(radio_map, rt.PlanarRadioMap):
        struct = get_or_add_planar_radio_map_mesh(
            name, radio_map, ps_groups, use_alpha=use_alpha
        )

        has_buffer = True
        try:
            struct.get_quantity_buffer(name, "colors")
        except ValueError:
            has_buffer = False

        # Prepare color-mapped radio map (directly on device)
        dr.eval(radio_map.path_gain)  # Note: important to avoid kernel misses.
        rm_values = dr.max(radio_map.path_gain, axis=0)
        texture, alpha = radio_map_texture(
            rm_values,
            db_scale=True,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            premultiply_alpha=use_alpha,
            rm_cmap=cfg.color_map,
        )

        if not has_buffer or not direct_update_from_device:
            # --- Register the radio map's texture & alpha buffers
            struct.add_color_quantity(
                name,
                texture.numpy().astype(np.float32),
                defined_on="texture",
                param_name="uv",
                enabled=True,
                image_origin="lower_left",
                filter_mode="nearest",
            )

            if use_alpha:
                # Note: texture-space alpha is not supported yet.
                struct.add_scalar_quantity(
                    f"{name}_alpha",
                    alpha.numpy().ravel().astype(np.float32),
                    defined_on="vertices",
                    enabled=False,
                )
                struct.set_transparency_quantity(f"{name}_alpha")

        else:
            # --- Update the existing buffers directly from the GPU
            # TODO: fix direct alpha updates and re-enable it.
            assert (
                not use_alpha
            ), "Alpha is not supported when updating directly from the device."

            rm_texture_buffer = struct.get_quantity_buffer(name, "colors")

            # TODO: figure out the weird component size mismatch so that we no longer need this padding
            texture_dr = dr.concat(
                [texture, dr.zeros(mi.TensorXf, shape=(*texture.shape[:-1], 1))],
                axis=-1,
            )

            rm_texture_buffer.update_data_from_device(texture_dr)
            if use_alpha:
                alphas_dr = mi.Float(alpha.ravel())
                rm_alpha_buffer = struct.get_quantity_buffer(name + "_alpha", "values")
                rm_alpha_buffer.update_data_from_device(alphas_dr)

    elif isinstance(radio_map, rt.MeshRadioMap):
        raise NotImplementedError("Mesh radio maps are not supported yet")
    else:
        raise ValueError(f"Unsupported radio map type: {type(radio_map)}")


def get_or_add_planar_radio_map_mesh(
    name: str,
    radio_map: rt.PlanarRadioMap,
    ps_groups: dict[str, ps.Group],
    use_alpha: bool = True,
) -> ps.SurfaceMesh:

    rm_shape = radio_map.path_gain.shape[1:]

    if ps.has_surface_mesh(name):
        struct = ps.get_surface_mesh(name)

        n_entries = dr.prod(rm_shape)
        if n_entries == struct.n_vertices():
            # TODO(!): need to update the vertices if pose changed, too
            return struct

    # Create rectangle mesh to display the planar radio map.
    if use_alpha:
        # We need one vertex per entry in the radio map because spatially-varying
        # transparency cannot be defined in texture space.
        vertices_x, vertices_y = np.meshgrid(
            np.linspace(-1, 1, rm_shape[1]),
            np.linspace(-1, 1, rm_shape[0]),
            indexing="xy",
        )
        vertices = np.stack(
            [
                vertices_x,
                vertices_y,
                np.zeros_like(vertices_x),
                np.ones_like(vertices_x),
            ],
            axis=-1,
        ).reshape(-1, 4)
        # Transform vertices to world coordinates (accounts from plane pose)
        to_world = radio_map.to_world.matrix.numpy().squeeze()
        vertices = (vertices @ to_world.T)[:, :3]

        # Faces: two triangles per cell
        # Adapted from: https://stackoverflow.com/a/44935368
        r = np.arange(vertices.shape[0]).reshape(rm_shape)
        faces = np.empty((rm_shape[0] - 1, rm_shape[1] - 1, 2, 3), dtype=int)
        faces[:, :, 0, 0] = r[:-1, :-1]
        faces[:, :, 1, 0] = r[:-1, 1:]
        faces[:, :, 0, 1] = r[:-1, 1:]
        faces[:, :, 1, 1] = r[1:, 1:]
        faces[:, :, :, 2] = r[1:, :-1, None]
        faces = faces.reshape(-1, 3)

    else:
        vertices = np.array(
            [
                [-1, -1, 0, 1],
                [1, -1, 0, 1],
                [1, 1, 0, 1],
                [-1, 1, 0, 1],
            ]
        )
        vertices_x = vertices[:, 0]
        vertices_y = vertices[:, 1]
        # Transform vertices to world coordinates (accounts from plane pose)
        to_world = radio_map.to_world.matrix.numpy().squeeze()
        vertices = (vertices @ to_world.T)[:, :3]

        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )

    # Add plane mesh to Polyscope
    struct = ps.register_surface_mesh(name, vertices=vertices, faces=faces)
    struct.add_to_group(ps_groups["radio_maps"])
    struct.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME, True)

    # UV map
    param_vals = np.stack(
        [
            (vertices_x.flatten() + 1) * 0.5,
            (vertices_y.flatten() + 1) * 0.5,
        ],
        axis=-1,
    )
    struct.add_parameterization_quantity(
        "uv", param_vals, defined_on="vertices", enabled=False
    )

    return struct


def add_paths_to_polyscope(
    gui: "SionnaRtGui", paths: rt.Paths | None, ps_groups: dict[str, ps.Group]
):
    if paths is None:
        return

    result = rt.render.paths_to_segments(paths)
    if not result:
        if ps.has_curve_network("paths"):
            ps.remove_curve_network("paths")
        return

    starts, ends, colors = result
    vertices = np.stack([starts, ends], axis=1).reshape(-1, 3)

    # TODO: appropriate colors for each path type
    # TODO: path transparency based on gain at each segment?
    struct = ps.register_curve_network(
        "paths",
        vertices,
        edges="segments",
        enabled=True,
    )
    display_radius = max(0.0001 * scene_scale(gui.scene), 0.3)
    struct.set_radius(display_radius, relative=False)

    struct.add_color_quantity(
        "path_colors",
        np.array(colors),
        defined_on="edges",
        enabled=True,
    )
    struct.set_transparency(0.6)
    struct.add_to_group(ps_groups["paths"])
    struct.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME, True)


def get_point_from_camera_center_ray(
    scene: rt.Scene,
    camera_to_world: np.ndarray,
) -> np.ndarray | None:

    ray = mi.Ray3f(camera_to_world[:3, 3], -camera_to_world[:3, 2])
    si = scene.mi_scene.ray_intersect(ray)
    if si.is_valid().numpy().item():
        return si.p.numpy().squeeze()

    return None


def get_normal_for_path(
    scene: rt.Scene,
    origin: np.ndarray,
    destination: np.ndarray,
) -> np.ndarray | None:

    ray = mi.Ray3f(origin, dr.normalize(destination - origin))

    si = scene.mi_scene.ray_intersect(ray)
    if si.is_valid().numpy().item():
        return si.n.numpy().squeeze()

    return None


def prepare_valid_taps(
    a: np.ndarray, tau: np.ndarray, tx_idx: int, return_vlims: bool = False
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """
    Inputs:
    - a: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    - tau: [num_rx, num_tx, num_paths] or [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]

    Outputs:
    - filtered_a: list of `num_rx` arrays of shape [num_paths]
    - filtered_tau: list of `num_rx` arrays of shape [num_paths]
    - n_valid: total number of valid taps
    - (min_a, max_a): optional
    - (min_tau, max_tau): optional
    """

    if tau.ndim == 3:
        # Reindex tau to match shape of a
        tau = tau[:, np.newaxis, :, np.newaxis, :, np.newaxis]

    # For each rx, filter out invalid taps
    filtered_a = []
    filtered_tau = []
    n_valid = 0
    tau_min, tau_max = np.inf, -np.inf
    a_min, a_max = np.inf, -np.inf
    for rx_idx in range(tau.shape[0]):
        valid = tau[rx_idx, :, tx_idx, ...] >= 0
        a_abs = np.abs(a[rx_idx, :, tx_idx, ...][valid])
        # Scale to ns
        tau_rescaled = tau[rx_idx, :, tx_idx, ...][valid] / 1e-9

        filtered_a.append(a_abs)
        filtered_tau.append(tau_rescaled)
        n_valid += tau_rescaled.size
        if return_vlims and tau_rescaled.size > 0:
            tau_min = min(tau_min, tau_rescaled.min())
            tau_max = max(tau_max, tau_rescaled.max())
            a_min = min(a_min, a_abs.min())
            a_max = max(a_max, a_abs.max())

    if return_vlims:
        return filtered_a, filtered_tau, n_valid, (a_min, a_max), (tau_min, tau_max)
    return filtered_a, filtered_tau, n_valid


def compute_thermal_noise_power(
    bandwidth_hz: float, temperature_k: float = 290.0
) -> float:
    """Compute thermal noise power in Watts for given bandwidth at temperature."""
    return K_BOLTZMANN * temperature_k * bandwidth_hz


def prepare_and_normalize_cir(
    taps: np.ndarray,
    tx_index: int,
    rx_index: int,
    num_taps: int,
    snr_offset_db: float,
    max_noise_std: float,
) -> dict:
    """Prepare a CIR message for the cir_zmq server.

    The output format matches the cir_zmq protocol:
      - taps are normalized (divided by their L2 norm per symbol)
      - norms carry the original channel gain for noise computation
      - sigma_scaling and sigma_max are global noise scalars
      - all arrays are flat 1D
    """
    # Input shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1]
    taps = taps[rx_index, 0, tx_index, 0, ...]  # [S, num_delay_bins] complex

    # Retain only the num_taps largest absolute taps per symbol
    tap_indices = np.argsort(np.abs(taps), axis=1)[:, ::-1][:, :num_taps]
    taps = np.take_along_axis(taps, tap_indices, axis=1)  # [S, T] complex

    # Convert to interleaved real/imag: [S, T] complex -> [S, 2*T] float
    taps_ri = np.stack([taps.real, taps.imag], axis=-1).reshape(taps.shape[0], -1)

    # Compute norms (L2 norm per symbol) and normalize taps
    norms = np.sqrt(np.sum(taps_ri**2, axis=-1))  # [S]
    nonzero = norms > 0
    taps_ri[nonzero] = taps_ri[nonzero] / norms[nonzero, None]

    # Compute sigma_scaling and sigma_max (global scalars)
    # Consistent with srk_batch_export.py: sigma_scaling is a pure dB offset factor.
    sigma_scaling = float(10.0 ** (-snr_offset_db / 20.0))
    sigma_max = float(max_noise_std)

    return {
        "msg_type": "cir",
        "sigma_scaling": sigma_scaling,
        "sigma_max": sigma_max,
        "norms": norms.astype(np.float32).flatten().tolist(),
        "taps": taps_ri.astype(np.float32).flatten().tolist(),
        "tap_indices": tap_indices.astype(np.uint16).flatten().tolist(),
    }


def prepare_and_normalize_cir_batch(
    taps: np.ndarray,
    tx_index: np.ndarray,
    rx_index: np.ndarray,
    num_taps: int,
    bandwidth: float,
    snr_offset_db: float,
    max_noise_std: float,
) -> dict:
    # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1]
    taps = taps[rx_index, 0, tx_index, 0, ...]
    num_time_steps = taps.shape[-2]
    # Retain only the num_taps largest absolute taps at each time step
    tap_indices = np.argsort(np.abs(taps), axis=-1)[..., ::-1][..., :num_taps]
    taps = np.take_along_axis(taps, tap_indices, axis=-1)

    # Convert to real/imag compatible with the C complex type
    taps = np.reshape(
        np.stack([taps.real, taps.imag], axis=-1),
        [len(rx_index), len(tx_index), num_time_steps, -1],
    )
    tap_indices = tap_indices.reshape(
        len(rx_index), len(tx_index), num_time_steps, num_taps
    )

    taps, norm, noise_std = _apply_noise_and_normalize(
        taps, bandwidth, snr_offset_db, max_noise_std
    )

    return {
        "msg_type": "cir",
        "taps": taps,
        "tap_indices": tap_indices.astype(np.uint16),
        "taps_norm": norm,
        "noise_std": noise_std,
    }


def _apply_noise_and_normalize(
    taps: np.ndarray, bandwidth: float, snr_offset_db: float, max_noise_std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Thermal noise power (kTB) in Watts at 290 K
    thermal_noise_power_w = compute_thermal_noise_power(bandwidth)
    noise_std = float(np.sqrt(thermal_noise_power_w))

    # Normalize taps and compute scaling norm
    norm = np.sqrt(np.sum(taps**2, axis=-1))
    zero_norm = norm == 0
    taps = np.where(zero_norm[..., None], taps, taps / norm[..., None])
    noise_std = np.where(zero_norm, 0, noise_std / norm[..., None])

    snr_offset_factor = 10 ** (-snr_offset_db / 20.0)
    noise_std *= snr_offset_factor

    taps = taps.astype(np.float32)
    noise_std = np.minimum(noise_std, max_noise_std).astype(np.float32)

    return taps, norm, noise_std

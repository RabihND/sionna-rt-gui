#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
3D visualization of antenna radiation patterns, following the same
construction as ``sionna.rt.AntennaPattern.show()``: the surface radius is
the linear gain, the color is the gain in dB (turbo colormap).
"""
from __future__ import annotations

import drjit as dr
import mitsuba as mi
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt
from sionna.rt.utils.geometry import rotation_matrix
from sionna.rt.utils.render import scene_scale

from .config import DEFAULT_SLICE_PLANE_NAME
from .ps_utils import ACCENT_BRIGHT, im_col32

PATTERN_STRUCTURE_NAME = "Antenna pattern"


# Dynamic range used for the dB radius scale and the color mapping
PATTERN_DB_RANGE = 40.0


def build_pattern_mesh(
    pattern, db_radius: bool = False, n: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample an antenna pattern (one polarization direction) on a sphere grid.
    Returns unit-scale vertices (max radius 1), triangle faces, and the
    per-vertex gain in dB. With `db_radius`, the radius is proportional to
    the gain in dB over a fixed dynamic range (like MATLAB's pattern()),
    which makes weak side lobes visible; otherwise the radius is the linear
    gain, as in sionna.rt's AntennaPattern.show().
    """
    theta = dr.linspace(mi.Float, 0, dr.pi, n, False)
    phi = dr.linspace(mi.Float, -dr.pi, dr.pi, n, False)
    theta_grid, phi_grid = dr.meshgrid(theta, phi, indexing="ij")
    c_theta, c_phi = pattern(theta_grid, phi_grid)

    theta_grid = np.reshape(theta_grid, [n, n])
    phi_grid = np.reshape(phi_grid, [n, n])
    c_theta = np.reshape(c_theta, [n, n])
    c_phi = np.reshape(c_phi, [n, n])

    gain = np.abs(c_theta) ** 2 + np.abs(c_phi) ** 2
    gain_db = 10 * np.log10(np.maximum(gain, 1e-5))

    if db_radius:
        db_max = float(np.max(gain_db))
        radius = np.clip(
            (gain_db - (db_max - PATTERN_DB_RANGE)) / PATTERN_DB_RANGE, 0.0, 1.0
        )
    else:
        # Linear gain, normalized so that the mesh fits in a unit sphere
        radius = gain / max(float(np.max(gain)), 1e-12)
    x = radius * np.sin(theta_grid) * np.cos(phi_grid)
    y = radius * np.sin(theta_grid) * np.sin(phi_grid)
    z = radius * np.cos(theta_grid)
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Quad grid faces, wrapping around in the phi direction
    index = np.arange(n * n).reshape(n, n)
    rolled = np.roll(index, -1, axis=1)
    i0 = index[:-1, :]
    i1 = rolled[:-1, :]
    i2 = rolled[1:, :]
    i3 = index[1:, :]
    quads = np.stack([i0, i1, i2, i3], axis=-1).reshape(-1, 4)
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)

    return vertices, faces, gain_db.reshape(-1)


def default_pattern_scale(gui: "SionnaRtGui") -> float:
    """Default display size of the pattern surface, in meters."""
    return max(0.02 * scene_scale(gui.scene), 12.0)


def remove_antenna_pattern_structure() -> None:
    if ps.has_surface_mesh(PATTERN_STRUCTURE_NAME):
        ps.get_surface_mesh(PATTERN_STRUCTURE_NAME).remove()


def update_antenna_pattern_structure(
    gui: "SionnaRtGui",
    rd: rt.RadioDevice,
    array: rt.AntennaArray,
    enabled: bool,
) -> None:
    """
    Show / hide / update the radiation pattern surface of the given device's
    antenna. The mesh is built once per antenna pattern and then follows the
    device through its transform, so per-frame updates stay cheap.
    """
    if not enabled:
        remove_antenna_pattern_structure()
        gui.antenna_pattern_cache_key = None
        return

    antenna_pattern = array.antenna_pattern
    db_radius = getattr(gui, "antenna_pattern_db_radius", False)
    cache_key = (id(antenna_pattern), db_radius)

    if (
        not ps.has_surface_mesh(PATTERN_STRUCTURE_NAME)
        or getattr(gui, "antenna_pattern_cache_key", None) != cache_key
    ):
        # Polarization direction 0, as in AntennaPattern.show()
        vertices, faces, gain_db = build_pattern_mesh(
            antenna_pattern[0], db_radius=db_radius
        )
        struct = ps.register_surface_mesh(
            PATTERN_STRUCTURE_NAME, vertices, faces, smooth_shade=True
        )
        # Fixed color range (strongest lobe down to -40 dB), so colors read
        # like MATLAB's pattern(): red at the main lobe, blue at weak lobes.
        db_max = float(np.max(gain_db))
        struct.add_scalar_quantity(
            "Gain [dB]",
            gain_db,
            cmap="turbo",
            vminmax=(db_max - PATTERN_DB_RANGE, db_max),
            enabled=True,
        )
        struct.set_transparency(0.75)
        struct.set_back_face_policy("identical")
        struct.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME, True)
        gui.antenna_pattern_cache_key = cache_key
    else:
        struct = ps.get_surface_mesh(PATTERN_STRUCTURE_NAME)

    # Follow the device: rotation + translation, scaled to the user-set size
    scale = getattr(gui, "antenna_pattern_scale", 0.0)
    if scale <= 0.0:
        scale = default_pattern_scale(gui)
        gui.antenna_pattern_scale = scale
    to_world = np.eye(4)
    to_world[:3, :3] = scale * rotation_matrix(rd.orientation).numpy()[..., 0]
    to_world[:3, 3] = rd.position.numpy().squeeze()
    struct.set_transform(to_world)


# ------------------------
# 2D polar cuts, equivalent to the vertical / horizontal cut figures
# produced by sionna.rt.AntennaPattern.show().


def compute_pattern_cuts(
    pattern,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample the vertical cut (full circle through the zenith, in the phi=0 /
    phi=pi plane) and the horizontal cut (theta=pi/2). Returns
    (vertical_angles, vertical_db, horizontal_angles, horizontal_db).
    """
    n = 361

    angles_v = np.linspace(0.0, 2.0 * np.pi, n)
    theta = np.where(angles_v <= np.pi, angles_v, 2.0 * np.pi - angles_v)
    phi = np.where(angles_v <= np.pi, 0.0, np.pi)
    c_theta, c_phi = [c.numpy() for c in pattern(mi.Float(theta), mi.Float(phi))]
    gain = np.abs(c_theta) ** 2 + np.abs(c_phi) ** 2
    v_db = 10.0 * np.log10(np.maximum(gain, 1e-6))

    angles_h = np.linspace(-np.pi, np.pi, n)
    c_theta, c_phi = [
        c.numpy()
        for c in pattern(mi.Float(np.full(n, 0.5 * np.pi)), mi.Float(angles_h))
    ]
    gain = np.abs(c_theta) ** 2 + np.abs(c_phi) ** 2
    h_db = 10.0 * np.log10(np.maximum(gain, 1e-6))

    return angles_v, v_db, angles_h, h_db


def _draw_polar_chart(
    draw_list,
    origin: tuple[float, float],
    size: float,
    ui_scale: float,
    title: str,
    angles: np.ndarray,
    values_db: np.ndarray,
    angle_to_xy,
    db_max: float,
    db_floor: float,
) -> None:
    """
    Draw one polar gain chart with ImDrawList primitives: dB grid rings,
    radial lines every 30 degrees, and the gain curve.
    """
    text_col = im_col32(0.78, 0.80, 0.81, 1.0)
    faint_text_col = im_col32(0.62, 0.64, 0.65, 0.85)
    grid_col = im_col32(0.55, 0.57, 0.58, 0.28)

    x0, y0 = origin
    draw_list.AddText((x0 + 4 * ui_scale, y0), text_col, title)

    cx = x0 + 0.5 * size
    cy = y0 + 0.5 * size + 8 * ui_scale
    radius = 0.5 * size - 28 * ui_scale

    # dB rings
    n_rings = 4
    for i in range(1, n_rings + 1):
        frac = i / n_rings
        draw_list.AddCircle((cx, cy), frac * radius, grid_col, 96)
        ring_db = db_floor + frac * (db_max - db_floor)
        draw_list.AddText(
            (cx + 3 * ui_scale, cy - frac * radius + 1 * ui_scale),
            faint_text_col,
            f"{ring_db:.0f}",
        )

    # Radial lines every 30 degrees, angle labels every 90
    for k in range(12):
        a = k * np.pi / 6.0
        dx, dy = angle_to_xy(np.array(a))
        draw_list.AddLine(
            (cx, cy), (cx + radius * float(dx), cy + radius * float(dy)), grid_col, 1.0
        )
        if k % 3 == 0:
            label_r = radius + 13 * ui_scale
            draw_list.AddText(
                (
                    cx + label_r * float(dx) - 9 * ui_scale,
                    cy + label_r * float(dy) - 7 * ui_scale,
                ),
                faint_text_col,
                f"{int(np.degrees(a))}",
            )

    # Gain curve
    span = max(db_max - db_floor, 1e-6)
    fracs = np.clip((values_db - db_floor) / span, 0.0, 1.0)
    dx, dy = angle_to_xy(angles)
    points = np.stack(
        [cx + radius * fracs * dx, cy + radius * fracs * dy], axis=1
    ).astype(np.float32)
    draw_list.AddPolyline(
        np.asfortranarray(points), im_col32(*ACCENT_BRIGHT), 0, 2.0 * ui_scale
    )


def pattern_cuts_window(gui: "SionnaRtGui", array: rt.AntennaArray) -> None:
    """
    Separate window with the vertical and horizontal polar cuts of the
    antenna gain, computed once per antenna pattern and drawn every frame.
    """
    if not getattr(gui, "show_pattern_cuts", False):
        return

    antenna_pattern = array.antenna_pattern
    key = id(antenna_pattern)
    cache = getattr(gui, "pattern_cuts_cache", None)
    if cache is None or cache["key"] != key:
        angles_v, v_db, angles_h, h_db = compute_pattern_cuts(antenna_pattern[0])
        db_max = float(np.ceil(max(np.max(v_db), np.max(h_db))))
        cache = {
            "key": key,
            "angles_v": angles_v,
            "v_db": v_db,
            "angles_h": angles_h,
            "h_db": h_db,
            "db_max": db_max,
            "db_floor": db_max - 40.0,
        }
        gui.pattern_cuts_cache = cache

    s = gui.ui_scale
    psim.SetNextWindowSize((720 * s, 420 * s), psim.ImGuiCond_FirstUseEver)
    psim.SetNextWindowPos((450 * s, 40 * s), psim.ImGuiCond_FirstUseEver)
    _, keep_open = psim.Begin("Antenna pattern cuts###pattern_cuts", open=True)
    if not keep_open:
        gui.show_pattern_cuts = False
        psim.End()
        return

    psim.TextDisabled(
        "Gain [dB] of a single antenna element (polarization direction 0)."
    )

    avail = psim.GetContentRegionAvail()
    chart = min(0.5 * (avail[0] - 16 * s), avail[1] - 4 * s)
    chart = max(chart, 120 * s)
    draw_list = psim.GetWindowDrawList()
    origin = psim.GetCursorScreenPos()

    # Vertical cut: 0 deg at the zenith, clockwise (as in AntennaPattern.show)
    _draw_polar_chart(
        draw_list,
        origin,
        chart,
        s,
        "Vertical cut G(theta, 0)",
        cache["angles_v"],
        cache["v_db"],
        lambda a: (np.sin(a), -np.cos(a)),
        cache["db_max"],
        cache["db_floor"],
    )
    # Horizontal cut: 0 deg to the right (east), counter-clockwise
    _draw_polar_chart(
        draw_list,
        (origin[0] + chart + 16 * s, origin[1]),
        chart,
        s,
        "Horizontal cut G(pi/2, phi)",
        cache["angles_h"],
        cache["h_db"],
        lambda a: (np.cos(a), -np.sin(a)),
        cache["db_max"],
        cache["db_floor"],
    )
    psim.Dummy((2 * chart + 16 * s, chart))
    psim.End()

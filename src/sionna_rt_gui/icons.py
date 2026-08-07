#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Vector icons drawn with ImGui's draw list.

Drawing them rather than shipping an icon font keeps the package free of binary
assets and their licenses, scales cleanly with the interface scale, and lets an
icon carry meaning specific to this application (a transmitter radiating, a
propagation path bouncing off a wall).

Each function draws inside the square (x, y) .. (x + size, y + size).
"""
from __future__ import annotations

import numpy as np
import polyscope.imgui as psim

from .ps_utils import im_col32


def _polyline(draw_list, points, color, thickness=1.6, closed=False):
    array = np.asarray(points, dtype=np.float32)
    draw_list.AddPolyline(
        np.asfortranarray(array),
        color,
        psim.ImDrawFlags_Closed if closed else 0,
        thickness,
    )


def transmitter(draw_list, x, y, size, color, scale=1.0):
    """
    A mast radiating: a vertical stem with expanding arcs. The arcs are centred
    low enough that the widest one still fits inside the icon's square.
    """
    cx = x + size * 0.5
    base = y + size * 0.88
    top = y + size * 0.34
    draw_list.AddLine((cx, base), (cx, top), color, 1.8 * scale)
    draw_list.AddCircleFilled((cx, top), size * 0.07, color)
    for i, radius in enumerate((0.14, 0.22, 0.30)):
        alpha = 1.0 - 0.22 * i
        faded = (color & 0x00FFFFFF) | (int(255 * alpha) << 24)
        draw_list.AddCircle((cx, top), size * radius, faded, 24, 1.4 * scale)
    draw_list.AddLine(
        (cx - size * 0.15, base), (cx + size * 0.15, base), color, 1.8 * scale
    )


def receiver(draw_list, x, y, size, color, scale=1.0):
    """A handset with a screen, and arcs arriving at its top corner."""
    left, right = x + size * 0.16, x + size * 0.52
    top, bottom = y + size * 0.32, y + size * 0.88
    draw_list.AddRect((left, top), (right, bottom), color, size * 0.07, 0, 1.6 * scale)
    # Screen
    draw_list.AddRect(
        (left + size * 0.05, top + size * 0.06),
        (right - size * 0.05, bottom - size * 0.12),
        (color & 0x00FFFFFF) | (130 << 24),
        size * 0.02,
        0,
        1.2 * scale,
    )
    # Home button
    draw_list.AddCircleFilled(
        (0.5 * (left + right), bottom - size * 0.06), size * 0.028, color
    )
    # Incoming signal: arcs centred on the top right corner, sized to stay
    # within the icon's square
    corner = (right - size * 0.02, top + size * 0.04)
    angles = np.linspace(-np.pi * 0.52, 0.02, 14)
    for i, radius in enumerate((0.14, 0.23, 0.32)):
        alpha = 1.0 - 0.2 * i
        arc = [
            (corner[0] + size * radius * np.cos(a), corner[1] + size * radius * np.sin(a))
            for a in angles
        ]
        _polyline(
            draw_list, arc, (color & 0x00FFFFFF) | (int(255 * alpha) << 24), 1.4 * scale
        )


def car(draw_list, x, y, size, color, scale=1.0):
    """A side profile with a cabin step and two wheels."""
    body_top, body_bottom = y + size * 0.50, y + size * 0.70
    _polyline(
        draw_list,
        [
            (x + size * 0.12, body_bottom),
            (x + size * 0.12, body_top),
            (x + size * 0.32, body_top),
            (x + size * 0.42, y + size * 0.36),
            (x + size * 0.66, y + size * 0.36),
            (x + size * 0.74, body_top),
            (x + size * 0.88, body_top),
            (x + size * 0.88, body_bottom),
        ],
        color,
        1.6 * scale,
    )
    for cx in (x + size * 0.30, x + size * 0.70):
        draw_list.AddCircle((cx, body_bottom), size * 0.10, color, 16, 1.6 * scale)


def person(draw_list, x, y, size, color, scale=1.0):
    """Head, body, arms and legs."""
    cx = x + size * 0.5
    draw_list.AddCircle((cx, y + size * 0.26), size * 0.11, color, 16, 1.6 * scale)
    draw_list.AddLine((cx, y + size * 0.37), (cx, y + size * 0.62), color, 1.6 * scale)
    draw_list.AddLine(
        (x + size * 0.30, y + size * 0.46),
        (x + size * 0.70, y + size * 0.46),
        color,
        1.6 * scale,
    )
    draw_list.AddLine(
        (cx, y + size * 0.62), (x + size * 0.34, y + size * 0.84), color, 1.6 * scale
    )
    draw_list.AddLine(
        (cx, y + size * 0.62), (x + size * 0.66, y + size * 0.84), color, 1.6 * scale
    )


def drone(draw_list, x, y, size, color, scale=1.0):
    """A quadcopter seen from above: body, arms and four rotors."""
    cx, cy = x + size * 0.5, y + size * 0.54
    reach = size * 0.26
    draw_list.AddRect(
        (cx - size * 0.10, cy - size * 0.08),
        (cx + size * 0.10, cy + size * 0.08),
        color,
        size * 0.03,
        0,
        1.5 * scale,
    )
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            px, py = cx + sx * reach, cy + sy * reach
            draw_list.AddLine((cx, cy), (px, py), color, 1.3 * scale)
            draw_list.AddCircle((px, py), size * 0.10, color, 14, 1.4 * scale)


def tree(draw_list, x, y, size, color, scale=1.0):
    """A trunk with a lobed canopy, and a hint of branches."""
    cx = x + size * 0.5
    trunk_top = y + size * 0.60
    draw_list.AddLine((cx, y + size * 0.88), (cx, trunk_top), color, 1.9 * scale)
    # Branches
    draw_list.AddLine(
        (cx, y + size * 0.70), (x + size * 0.34, y + size * 0.58), color, 1.3 * scale
    )
    draw_list.AddLine(
        (cx, y + size * 0.74), (x + size * 0.66, y + size * 0.62), color, 1.3 * scale
    )
    # Canopy: three overlapping lobes read as foliage where a circle does not
    for offset_x, offset_y, radius in (
        (-0.15, 0.44, 0.15),
        (0.15, 0.44, 0.15),
        (0.0, 0.30, 0.18),
    ):
        draw_list.AddCircle(
            (cx + size * offset_x, y + size * offset_y),
            size * radius,
            color,
            18,
            1.5 * scale,
        )


def wall(draw_list, x, y, size, color, scale=1.0):
    """Brick courses."""
    left, right = x + size * 0.18, x + size * 0.82
    for i in range(3):
        top = y + size * (0.34 + 0.17 * i)
        draw_list.AddRect(
            (left, top), (right, top + size * 0.15), color, 0.0, 0, 1.4 * scale
        )
        draw_list.AddLine(
            (left + size * (0.32 if i % 2 else 0.16), top),
            (left + size * (0.32 if i % 2 else 0.16), top + size * 0.15),
            color,
            1.2 * scale,
        )


def glass(draw_list, x, y, size, color, scale=1.0):
    """A pane with a highlight."""
    left, right = x + size * 0.24, x + size * 0.76
    top, bottom = y + size * 0.26, y + size * 0.82
    draw_list.AddRect((left, top), (right, bottom), color, 0.0, 0, 1.6 * scale)
    draw_list.AddLine(
        (left + size * 0.06, bottom - size * 0.06),
        (right - size * 0.06, top + size * 0.06),
        (color & 0x00FFFFFF) | (150 << 24),
        1.3 * scale,
    )


def radio_map(draw_list, x, y, size, color, scale=1.0):
    """A grid in perspective, like the map draped over the ground."""
    top_left = (x + size * 0.30, y + size * 0.34)
    top_right = (x + size * 0.70, y + size * 0.34)
    bottom_right = (x + size * 0.90, y + size * 0.78)
    bottom_left = (x + size * 0.10, y + size * 0.78)
    _polyline(
        draw_list,
        [top_left, top_right, bottom_right, bottom_left],
        color,
        1.5 * scale,
        closed=True,
    )
    for t in (0.33, 0.66):
        draw_list.AddLine(
            (
                top_left[0] + t * (top_right[0] - top_left[0]),
                top_left[1],
            ),
            (
                bottom_left[0] + t * (bottom_right[0] - bottom_left[0]),
                bottom_left[1],
            ),
            (color & 0x00FFFFFF) | (120 << 24),
            1.2 * scale,
        )
        draw_list.AddLine(
            (
                top_left[0] + t * (bottom_left[0] - top_left[0]),
                top_left[1] + t * (bottom_left[1] - top_left[1]),
            ),
            (
                top_right[0] + t * (bottom_right[0] - top_right[0]),
                top_right[1] + t * (bottom_right[1] - top_right[1]),
            ),
            (color & 0x00FFFFFF) | (120 << 24),
            1.2 * scale,
        )


def paths(draw_list, x, y, size, color, scale=1.0):
    """A ray bouncing off a surface between two endpoints."""
    start = (x + size * 0.16, y + size * 0.32)
    bounce = (x + size * 0.52, y + size * 0.74)
    end = (x + size * 0.86, y + size * 0.30)
    _polyline(draw_list, [start, bounce, end], color, 1.6 * scale)
    draw_list.AddCircleFilled(start, size * 0.075, color)
    draw_list.AddCircleFilled(end, size * 0.075, color)
    draw_list.AddLine(
        (x + size * 0.30, y + size * 0.84),
        (x + size * 0.78, y + size * 0.84),
        (color & 0x00FFFFFF) | (140 << 24),
        1.4 * scale,
    )


def antenna(draw_list, x, y, size, color, scale=1.0):
    """
    A polar pattern plot: dB rings with a lobe drawn on them, which is what the
    antenna pattern view actually shows.
    """
    cx, cy = x + size * 0.5, y + size * 0.52
    outer = size * 0.40
    # Rings and axes of the polar grid
    faint = (color & 0x00FFFFFF) | (110 << 24)
    for ring in (0.45, 0.75, 1.0):
        draw_list.AddCircle((cx, cy), outer * ring, faint, 22, 1.1 * scale)
    draw_list.AddLine((cx - outer, cy), (cx + outer, cy), faint, 1.0 * scale)
    draw_list.AddLine((cx, cy - outer), (cx, cy + outer), faint, 1.0 * scale)
    # A lobe pointing up, as on a vertical cut
    angles = np.linspace(0.0, 2.0 * np.pi, 40)
    radii = outer * (0.30 + 0.70 * np.clip(np.cos(angles - np.pi / 2), 0.0, 1.0) ** 1.6)
    _polyline(
        draw_list,
        [(cx + r * np.cos(a), cy + r * np.sin(a)) for a, r in zip(angles, radii)],
        color,
        1.6 * scale,
        closed=True,
    )


def antenna_lobe(draw_list, x, y, size, color, scale=1.0):
    """A main lobe with a small back lobe, for the 3D pattern."""
    cx, cy = x + size * 0.30, y + size * 0.54
    draw_list.AddCircleFilled((cx, cy), size * 0.055, color)
    # Main lobe: a cardioid pointing right, closed back at the origin
    angles = np.linspace(-np.pi / 2, np.pi / 2, 26)
    radii = size * 0.50 * np.cos(angles) ** 1.4
    lobe = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a, r in zip(angles, radii)]
    _polyline(draw_list, lobe + [(cx, cy)], color, 1.5 * scale)
    # Back lobe
    back = np.linspace(np.pi / 2, 3 * np.pi / 2, 16)
    back_radii = size * 0.16 * np.cos(back - np.pi) ** 2
    _polyline(
        draw_list,
        [(cx + r * np.cos(a), cy + r * np.sin(a)) for a, r in zip(back, back_radii)]
        + [(cx, cy)],
        (color & 0x00FFFFFF) | (150 << 24),
        1.3 * scale,
    )


def object_icon(draw_list, x, y, size, color, scale=1.0):
    """A cube, for the active object."""
    half = size * 0.24
    cx, cy = x + size * 0.5, y + size * 0.56
    top = [
        (cx, cy - half * 1.35),
        (cx + half, cy - half * 0.65),
        (cx, cy),
        (cx - half, cy - half * 0.65),
    ]
    _polyline(draw_list, top, color, 1.5 * scale, closed=True)
    for corner in (top[1], top[2], top[3]):
        draw_list.AddLine(
            corner, (corner[0], corner[1] + half * 1.25), color, 1.4 * scale
        )
    _polyline(
        draw_list,
        [
            (top[1][0], top[1][1] + half * 1.25),
            (top[2][0], top[2][1] + half * 1.25),
            (top[3][0], top[3][1] + half * 1.25),
        ],
        color,
        1.4 * scale,
    )


def scene_icon(draw_list, x, y, size, color, scale=1.0):
    """Two buildings on the ground."""
    ground = y + size * 0.80
    draw_list.AddRect(
        (x + size * 0.20, y + size * 0.42), (x + size * 0.45, ground), color, 0.0, 0, 1.5 * scale
    )
    draw_list.AddRect(
        (x + size * 0.52, y + size * 0.30), (x + size * 0.80, ground), color, 0.0, 0, 1.5 * scale
    )
    draw_list.AddLine(
        (x + size * 0.10, ground), (x + size * 0.90, ground), color, 1.5 * scale
    )


def render_icon(draw_list, x, y, size, color, scale=1.0):
    """A camera."""
    left, right = x + size * 0.20, x + size * 0.68
    top, bottom = y + size * 0.38, y + size * 0.72
    draw_list.AddRect((left, top), (right, bottom), color, size * 0.05, 0, 1.6 * scale)
    _polyline(
        draw_list,
        [
            (right, y + size * 0.48),
            (x + size * 0.86, y + size * 0.40),
            (x + size * 0.86, y + size * 0.70),
            (right, y + size * 0.62),
        ],
        color,
        1.5 * scale,
        closed=True,
    )


def timeline_icon(draw_list, x, y, size, color, scale=1.0):
    """A play triangle."""
    _polyline(
        draw_list,
        [
            (x + size * 0.34, y + size * 0.30),
            (x + size * 0.76, y + size * 0.54),
            (x + size * 0.34, y + size * 0.78),
        ],
        color,
        1.6 * scale,
        closed=True,
    )


def impulse_icon(draw_list, x, y, size, color, scale=1.0):
    """Stems of decreasing height, as in an impulse response."""
    base = y + size * 0.78
    for i, height in enumerate((0.42, 0.28, 0.34, 0.16, 0.10)):
        px = x + size * (0.20 + 0.15 * i)
        draw_list.AddLine((px, base), (px, base - size * height), color, 1.5 * scale)
        draw_list.AddCircleFilled((px, base - size * height), size * 0.045, color)
    draw_list.AddLine(
        (x + size * 0.14, base), (x + size * 0.88, base), color, 1.4 * scale
    )


def truck(draw_list, x, y, size, color, scale=1.0):
    """A box body with a lower cab in front, unmistakable against the car."""
    wheel_y = y + size * 0.72
    body_top = y + size * 0.26
    # Cargo box
    draw_list.AddRect(
        (x + size * 0.10, body_top),
        (x + size * 0.58, wheel_y),
        color,
        0.0,
        0,
        1.6 * scale,
    )
    # Cab, lower, with a sloped windscreen
    _polyline(
        draw_list,
        [
            (x + size * 0.58, y + size * 0.48),
            (x + size * 0.72, y + size * 0.48),
            (x + size * 0.88, y + size * 0.60),
            (x + size * 0.88, wheel_y),
            (x + size * 0.58, wheel_y),
        ],
        color,
        1.6 * scale,
    )
    for cx in (x + size * 0.24, x + size * 0.46, x + size * 0.76):
        draw_list.AddCircle((cx, wheel_y), size * 0.095, color, 14, 1.5 * scale)


ASSET_ICONS = {
    "car": car,
    "truck": truck,
    "human": person,
    "drone": drone,
    "tree": tree,
    "wall": wall,
    "glass": glass,
}


def icon_button(
    identifier: str,
    draw,
    size: float,
    scale: float,
    tooltip: str = "",
    active: bool = False,
    accent=None,
) -> bool:
    """A square button carrying a drawn icon. Returns True when pressed."""
    from .workspace_layout import SELECT_BLUE

    if active:
        psim.PushStyleColor(psim.ImGuiCol_Button, SELECT_BLUE)
        psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, SELECT_BLUE)
    position = psim.GetCursorScreenPos()
    pressed = psim.Button(f"##icon_{identifier}", (size, size))
    if active:
        psim.PopStyleColor(2)
    hovered = psim.IsItemHovered()
    if tooltip and hovered:
        psim.SetTooltip(tooltip)

    color = im_col32(*(accent if accent is not None else (0.88, 0.89, 0.90)))
    draw(
        psim.GetWindowDrawList(),
        position[0],
        position[1],
        size,
        color,
        scale,
    )
    return pressed

#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Library of ready-to-place scene assets (vehicles, people, drones, panels).

Each asset is a procedurally generated mesh plus a radio material chosen for
that kind of object, so a placed asset participates in propagation with
plausible electromagnetic properties. Geometry is deliberately simple: at
radio wavelengths (centimeters to decimeters) the gross shape and material
dominate, and coarse proxies are the usual practice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import drjit as dr
import mitsuba as mi
import numpy as np
from sionna import rt


@dataclass(frozen=True)
class AssetSpec:
    key: str
    label: str
    # Overall extents along x (length), y (width), z (height), in meters.
    # For the quadcopter, x is the motor-to-motor span.
    size: tuple[float, float, float]
    color: tuple[float, float, float]
    # Either ("itu", <itu_type>) or ("custom", <RadioMaterial kwargs>)
    material: tuple[str, object]
    note: str
    scattering_coefficient: float = 0.0
    # Height above the drop point, for objects that fly
    hover_height: float = 0.0
    # Material name for assets that are not covered by ITU-R P.2040
    custom_material_name: str = ""


# Tissue is not covered by ITU-R P.2040: these are typical values for muscle
# and skin between 2 and 6 GHz (relative permittivity ~40, conductivity
# ~2 S/m), i.e. a lossy dielectric that attenuates rather than reflects.
HUMAN_MATERIAL = {
    "relative_permittivity": 41.4,
    "conductivity": 1.9,
    "thickness": 0.3,
    "scattering_coefficient": 0.3,
}
# Foliage: low permittivity, mildly lossy, strongly scattering.
FOLIAGE_MATERIAL = {
    "relative_permittivity": 1.5,
    "conductivity": 0.05,
    "thickness": 1.0,
    "scattering_coefficient": 0.6,
}

ASSET_LIBRARY: tuple[AssetSpec, ...] = (
    AssetSpec(
        key="car",
        label="Car",
        size=(4.6, 1.85, 1.48),
        color=(0.72, 0.18, 0.18),
        material=("itu", "metal"),
        note="Sedan, metal body: a strong reflector and blocker.",
    ),
    AssetSpec(
        key="truck",
        label="Truck",
        size=(9.0, 2.55, 3.5),
        color=(0.32, 0.42, 0.68),
        material=("itu", "metal"),
        note="Box truck, metal: useful for blockage studies.",
    ),
    AssetSpec(
        key="human",
        label="Person",
        size=(0.5, 0.45, 1.75),
        color=(0.85, 0.65, 0.5),
        material=("custom", HUMAN_MATERIAL),
        custom_material_name="human-tissue",
        note=(
            "Tissue-like lossy dielectric (eps_r 41, sigma 1.9 S/m).\n"
            "Absorbs more than it reflects, as a body does."
        ),
        scattering_coefficient=0.3,
    ),
    AssetSpec(
        key="drone",
        label="Drone",
        # Motor-to-motor span in the DJI Matrice / Inspire class
        size=(0.85, 0.85, 0.35),
        color=(0.22, 0.22, 0.25),
        material=("itu", "metal"),
        note=(
            "Quadcopter, Matrice/Inspire class (0.85 m span).\n"
            "Metal body, motors and propellers."
        ),
        hover_height=2.0,
    ),
    AssetSpec(
        key="wall",
        label="Concrete wall",
        size=(6.0, 0.3, 3.0),
        color=(0.6, 0.6, 0.58),
        material=("itu", "concrete"),
        note="Concrete slab: blocks and attenuates.",
    ),
    AssetSpec(
        key="glass",
        label="Glass panel",
        size=(3.0, 0.1, 2.5),
        color=(0.55, 0.7, 0.75),
        material=("itu", "glass"),
        note="Window glass: partially transmits.",
    ),
    AssetSpec(
        key="tree",
        label="Tree",
        size=(5.0, 5.0, 8.0),
        color=(0.25, 0.45, 0.2),
        material=("custom", FOLIAGE_MATERIAL),
        custom_material_name="foliage",
        note="Trunk and canopy: weakly reflecting, strongly scattering.",
        scattering_coefficient=0.6,
    ),
)


def _mesh_from_arrays(name: str, vertices: np.ndarray, faces: np.ndarray) -> mi.Mesh:
    mesh = mi.Mesh(
        name,
        vertex_count=vertices.shape[0],
        face_count=faces.shape[0],
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )
    params = mi.traverse(mesh)
    params["vertex_positions"] = dr.ravel(mi.Point3f(vertices.T))
    params["faces"] = dr.ravel(mi.Vector3u(faces.T.astype(np.uint32)))
    params.update()
    return mesh


def _merge(parts: list[tuple[np.ndarray, np.ndarray]]):
    """Concatenate (vertices, faces) parts into a single mesh."""
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for part_vertices, part_faces in parts:
        vertices.append(part_vertices)
        faces.append(part_faces + offset)
        offset += part_vertices.shape[0]
    return (
        np.concatenate(vertices, axis=0).astype(np.float32),
        np.concatenate(faces, axis=0),
    )


def _tube(radius, length, center, axis: str = "z", sides: int = 12):
    """Closed cylinder of the given radius and length along one axis."""
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    circle = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    half = length / 2.0
    # Axis order so the ring lies in the plane perpendicular to `axis`
    order = {"x": (2, 0, 1), "y": (0, 2, 1), "z": (0, 1, 2)}[axis]

    def place(u, v, w):
        point = np.empty((len(u), 3))
        point[:, order[0]] = u
        point[:, order[1]] = v
        point[:, order[2]] = w
        return point

    low = place(circle[:, 0], circle[:, 1], np.full(sides, -half))
    high = place(circle[:, 0], circle[:, 1], np.full(sides, half))
    caps = place(np.zeros(2), np.zeros(2), np.array([-half, half]))
    vertices = np.concatenate([low, high, caps], axis=0) + np.asarray(center)
    low_center, high_center = 2 * sides, 2 * sides + 1

    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([i, j, sides + j])
        faces.append([i, sides + j, sides + i])
        faces.append([low_center, j, i])
        faces.append([high_center, sides + i, sides + j])
    return vertices, np.array(faces)


def _ellipsoid(radii, center, segments: int = 12, rings: int = 6):
    """UV sphere, optionally squashed per axis."""
    radii = np.asarray(radii, dtype=float)
    thetas = np.linspace(0.0, np.pi, rings + 1)[1:-1]
    phis = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring_points = []
    for theta in thetas:
        ring_points.append(
            np.stack(
                [
                    radii[0] * np.sin(theta) * np.cos(phis),
                    radii[1] * np.sin(theta) * np.sin(phis),
                    np.full(segments, radii[2] * np.cos(theta)),
                ],
                axis=1,
            )
        )
    body = np.concatenate(ring_points, axis=0)
    poles = np.array([[0.0, 0.0, radii[2]], [0.0, 0.0, -radii[2]]])
    vertices = np.concatenate([body, poles], axis=0) + np.asarray(center)
    top, bottom = body.shape[0], body.shape[0] + 1

    faces = []
    n_rings = len(thetas)
    for r in range(n_rings - 1):
        for i in range(segments):
            j = (i + 1) % segments
            a = r * segments + i
            b = r * segments + j
            c = (r + 1) * segments + j
            d = (r + 1) * segments + i
            faces.append([a, b, c])
            faces.append([a, c, d])
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([top, i, j])
        faces.append([bottom, (n_rings - 1) * segments + j, (n_rings - 1) * segments + i])
    return vertices, np.array(faces)


def _car_parts(size):
    """Sedan silhouette: lower body, inset cabin, four wheels."""
    length, width, height = size
    wheel_radius = 0.19 * height
    body_bottom = wheel_radius
    body_top = 0.62 * height
    parts = [
        _box_arrays(
            (length, width, body_top - body_bottom),
            (0.0, 0.0, 0.5 * (body_bottom + body_top)),
        ),
        # Cabin: shorter, narrower, set back a little
        _box_arrays(
            (0.5 * length, 0.86 * width, height - body_top),
            (-0.04 * length, 0.0, 0.5 * (body_top + height)),
        ),
    ]
    for dx in (0.31 * length, -0.31 * length):
        for dy in (0.45 * width, -0.45 * width):
            parts.append(
                _tube(wheel_radius, 0.1 * width, (dx, dy, wheel_radius), axis="y")
            )
    return _merge(parts)


def _person_parts(size):
    """Standing person: legs, torso, arms, head."""
    _, width, height = size
    head_radius = 0.058 * height
    leg_length = 0.47 * height
    torso_height = 0.34 * height
    parts = [
        # Legs
        _tube(0.055 * height, leg_length, (0.0, -0.055 * height, 0.5 * leg_length)),
        _tube(0.055 * height, leg_length, (0.0, 0.055 * height, 0.5 * leg_length)),
        # Torso (slightly flattened front-to-back)
        _ellipsoid(
            (0.5 * 0.62 * width, 0.5 * width, 0.5 * torso_height),
            (0.0, 0.0, leg_length + 0.5 * torso_height),
        ),
        # Arms, hanging just outside the torso
        _tube(
            0.032 * height,
            0.78 * torso_height,
            (0.0, -0.44 * width, leg_length + 0.52 * torso_height),
        ),
        _tube(
            0.032 * height,
            0.78 * torso_height,
            (0.0, 0.44 * width, leg_length + 0.52 * torso_height),
        ),
        # Neck and head
        _tube(0.028 * height, 0.05 * height, (0.0, 0.0, leg_length + torso_height)),
        _ellipsoid(
            (0.82 * head_radius, head_radius, 1.1 * head_radius),
            (0.0, 0.0, leg_length + torso_height + 0.05 * height + head_radius),
        ),
    ]
    return _merge(parts)


def _quadcopter_parts(size):
    """
    Quadcopter in the size class of a DJI Matrice / Inspire: central body,
    four arms at 45 degrees, motors, and propeller discs.
    """
    span = size[0]  # motor-to-motor diagonal
    arm_reach = 0.5 * span / np.sqrt(2.0)
    body_height = 0.10 * span
    propeller_radius = 0.23 * span
    parts = [
        # Fuselage and a gimbal/payload bump underneath
        _box_arrays((0.34 * span, 0.22 * span, body_height), (0.0, 0.0, 0.0)),
        _ellipsoid(
            (0.07 * span, 0.07 * span, 0.05 * span),
            (0.10 * span, 0.0, -0.5 * body_height - 0.03 * span),
        ),
    ]
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            x, y = sx * arm_reach, sy * arm_reach
            # Arm: a tube pointing outwards, approximated along its dominant axis
            parts.append(
                _tube(
                    0.022 * span,
                    abs(2 * x),
                    (0.5 * x, 0.5 * y, 0.0),
                    axis="x",
                )
            )
            parts.append(
                _tube(
                    0.022 * span,
                    abs(2 * y),
                    (x, 0.5 * y, 0.0),
                    axis="y",
                )
            )
            # Motor and propeller disc
            parts.append(_tube(0.035 * span, 0.07 * span, (x, y, 0.03 * span)))
            parts.append(
                _tube(propeller_radius, 0.008 * span, (x, y, 0.08 * span), sides=16)
            )
            # Landing leg
            parts.append(
                _tube(
                    0.015 * span,
                    0.22 * span,
                    (0.6 * x, 0.6 * y, -0.5 * body_height - 0.11 * span),
                )
            )
    return _merge(parts)


def _truck_parts(size):
    """Box truck: cab, cargo box, wheels."""
    length, width, height = size
    wheel_radius = 0.13 * height
    chassis = wheel_radius
    cab_length = 0.28 * length
    parts = [
        # Cargo box
        _box_arrays(
            (length - cab_length, width, height - chassis),
            (-0.5 * cab_length, 0.0, chassis + 0.5 * (height - chassis)),
        ),
        # Cab, a bit lower than the cargo box
        _box_arrays(
            (cab_length, width, 0.62 * (height - chassis)),
            (
                0.5 * (length - cab_length),
                0.0,
                chassis + 0.31 * (height - chassis),
            ),
        ),
    ]
    for dx in (0.36 * length, -0.12 * length, -0.34 * length):
        for dy in (0.44 * width, -0.44 * width):
            parts.append(
                _tube(wheel_radius, 0.12 * width, (dx, dy, wheel_radius), axis="y")
            )
    return _merge(parts)


def _tree_parts(size):
    """Trunk plus a rounded canopy."""
    width, _, height = size
    trunk_height = 0.42 * height
    canopy_radius = 0.5 * width
    parts = [
        _tube(0.045 * height, trunk_height, (0.0, 0.0, 0.5 * trunk_height), sides=10),
        _ellipsoid(
            (canopy_radius, canopy_radius, 0.5 * (height - trunk_height)),
            (0.0, 0.0, trunk_height + 0.5 * (height - trunk_height)),
            segments=14,
            rings=7,
        ),
    ]
    return _merge(parts)


def _panel_parts(size):
    # Standing on the ground, like a wall or a window pane
    return _box_arrays(size, (0.0, 0.0, 0.5 * size[2]))


PART_BUILDERS = {
    "car": _car_parts,
    "truck": _truck_parts,
    "human": _person_parts,
    "drone": _quadcopter_parts,
    "tree": _tree_parts,
    "wall": _panel_parts,
    "glass": _panel_parts,
}


def _box_arrays(size, center) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(size, dtype=np.float32) / 2.0
    # Corner i encodes the (x, y, z) choices in its bits
    corners = np.array(
        [
            [x, y, z]
            for x in (-half[0], half[0])
            for y in (-half[1], half[1])
            for z in (-half[2], half[2])
        ],
        dtype=np.float32,
    ) + np.asarray(center, dtype=np.float32)
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2],
            [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1],
            [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4],
            [1, 5, 7], [1, 7, 3],
        ]
    )
    return corners, faces


def build_asset_mesh(spec: AssetSpec, name: str, base_position) -> mi.Mesh:
    """
    Build the asset's mesh. Builders model the object in a local frame whose
    origin sits at the ground contact point, so the mesh is simply shifted to
    `base_position` (plus the asset's hover height, for flying objects).
    """
    builder = PART_BUILDERS.get(spec.key)
    if builder is None:
        vertices, faces = _box_arrays(spec.size, (0.0, 0.0, 0.5 * spec.size[2]))
    else:
        vertices, faces = builder(spec.size)
    offset = np.asarray(base_position, dtype=float) + [0.0, 0.0, spec.hover_height]
    return _mesh_from_arrays(name, vertices + offset, faces)


def asset_material_name(spec: AssetSpec) -> str:
    """Name the asset's material after what it is made of."""
    kind, value = spec.material
    return value if kind == "itu" else spec.custom_material_name


def build_asset_material(spec: AssetSpec, scene: rt.Scene | None = None):
    """
    Material for an asset. Materials are shared by name, so placing three cars
    reuses one "metal" material rather than adding a new one each time, and the
    scene's existing materials are reused when they match.
    """
    name = asset_material_name(spec)
    if scene is not None:
        existing = scene.radio_materials.get(name)
        if existing is not None:
            return existing

    kind, value = spec.material
    if kind == "itu":
        return rt.ITURadioMaterial(
            name=name,
            itu_type=value,
            thickness=0.05,
            scattering_coefficient=spec.scattering_coefficient,
            color=spec.color,
        )
    return rt.RadioMaterial(name=name, color=spec.color, **value)

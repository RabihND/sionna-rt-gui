#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Named cameras that can be pointed at, or carried by, something in the scene.

A camera holds a position and a target, as ``sionna.rt.Camera`` does, and adds
two switches on top of it:

- where it sits: free, carried by an object at a fixed offset, or circling a
  centre point
- where it aims: free, or locked onto an object so a moving device stays in
  frame

That covers a fly-over that follows a car, a fixed mast that turns to track a
receiver, and an orbit for showing a scene off, without a separate mode for
each. The viewer's own camera is driven from the active one, so anything that
reads the view - screenshots included - sees it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import TYPE_CHECKING

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt

from .ps_utils import ACCENT_BRIGHT
from .workspace_layout import (
    end_property_row,
    numeric_field,
    numeric_field3,
    property_row,
)

if TYPE_CHECKING:
    from .gui import SionnaRtGui

# Where a camera sits
PLACEMENT_FREE = "Free"
PLACEMENT_CARRIED = "Carried by object"
PLACEMENT_ORBIT = "Orbit a point"
PLACEMENTS = [PLACEMENT_FREE, PLACEMENT_CARRIED, PLACEMENT_ORBIT]

# Where it aims
AIM_FREE = "Free"
AIM_LOCKED = "Locked to object"
AIMS = [AIM_FREE, AIM_LOCKED]


@dataclass
class ViewCamera:
    """One named camera. Positions are in scene units, angles in degrees."""

    name: str
    position: np.ndarray = field(
        default_factory=lambda: np.array([100.0, 0.0, 80.0], dtype=np.float64)
    )
    target: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    placement: str = PLACEMENT_FREE
    aim: str = AIM_FREE

    # Carried by, and locked to: names of a radio device or scene object
    carrier: str | None = None
    # Offset kept from whatever carries the camera, e.g. straight up for a
    # fly-over
    offset: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -25.0, 40.0], dtype=np.float64)
    )
    locked_to: str | None = None

    # Orbit
    orbit_center: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    orbit_radius: float = 120.0
    orbit_height: float = 90.0
    orbit_speed_deg_s: float = 12.0
    orbit_angle_deg: float = 0.0

    def to_sionna(self) -> rt.Camera:
        """The same camera as sionna's own, for rendering a still of the scene."""
        camera = rt.Camera(position=[float(v) for v in self.position])
        camera.look_at([float(v) for v in self.target])
        return camera


def default_camera(gui: "SionnaRtGui", name: str) -> ViewCamera:
    """A camera placed on the current view, so it starts where the user is."""
    camera = ViewCamera(name=name)
    capture_current_view(gui, camera)
    return camera


def add_camera(gui: "SionnaRtGui") -> ViewCamera:
    """Add a camera on the current view and select it."""
    index = len(gui.cameras) + 1
    names = {camera.name for camera in gui.cameras}
    while f"camera-{index}" in names:
        index += 1
    camera = default_camera(gui, f"camera-{index}")
    gui.cameras.append(camera)
    return camera


def remove_camera(gui: "SionnaRtGui", camera: ViewCamera) -> None:
    if gui.active_camera is camera:
        gui.active_camera = None
    if gui.selected_camera is camera:
        gui.selected_camera = None
    if camera in gui.cameras:
        gui.cameras.remove(camera)


def capture_current_view(gui: "SionnaRtGui", camera: ViewCamera) -> None:
    """Copy the viewer's current view into the camera."""
    parameters = ps.get_view_camera_parameters()
    position = np.asarray(parameters.get_position(), dtype=np.float64)
    look_direction = np.asarray(parameters.get_look_dir(), dtype=np.float64)
    # Keep looking at whatever is in the middle of the view: the distance to the
    # scene centre is a reasonable stand-in for how far away that is
    distance = max(float(np.linalg.norm(gui.scene_center() - position)), 1.0)
    camera.position = position
    camera.target = position + look_direction * distance


def activate(gui: "SionnaRtGui", camera: ViewCamera | None) -> None:
    """Hand the view over to a camera, or back to the mouse."""
    gui.active_camera = camera
    if camera is not None:
        gui._camera_drive_time = time.time()
        apply_camera(gui, camera, 0.0)


def resolved_pose(
    gui: "SionnaRtGui", camera: ViewCamera, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Where the camera is and what it looks at this frame, after its placement
    and aim have had their say.
    """
    position = np.asarray(camera.position, dtype=np.float64).copy()
    target = np.asarray(camera.target, dtype=np.float64).copy()

    if camera.placement == PLACEMENT_ORBIT:
        camera.orbit_angle_deg = (
            camera.orbit_angle_deg + camera.orbit_speed_deg_s * dt
        ) % 360.0
        angle = math.radians(camera.orbit_angle_deg)
        centre = np.asarray(camera.orbit_center, dtype=np.float64)
        position = centre + np.array(
            [
                camera.orbit_radius * math.cos(angle),
                camera.orbit_radius * math.sin(angle),
                camera.orbit_height,
            ]
        )
        target = centre
    elif camera.placement == PLACEMENT_CARRIED and camera.carrier:
        carried_from = gui.look_at_target_position(camera.carrier)
        if carried_from is not None:
            position = np.asarray(carried_from, dtype=np.float64) + np.asarray(
                camera.offset, dtype=np.float64
            )
            # Something carrying the camera is worth looking at unless told
            # otherwise
            if camera.aim == AIM_FREE:
                target = np.asarray(carried_from, dtype=np.float64)

    if camera.aim == AIM_LOCKED and camera.locked_to:
        locked_position = gui.look_at_target_position(camera.locked_to)
        if locked_position is not None:
            target = np.asarray(locked_position, dtype=np.float64)

    return position, target


def apply_camera(gui: "SionnaRtGui", camera: ViewCamera, dt: float) -> None:
    """Point the viewer's camera where this one is looking."""
    position, target = resolved_pose(gui, camera, dt)
    if np.allclose(position, target, atol=1e-6):
        return
    ps.look_at(tuple(float(v) for v in position), tuple(float(v) for v in target))


def update_active_camera(gui: "SionnaRtGui") -> None:
    """Drive the view from the active camera, once per frame."""
    camera = gui.active_camera
    if camera is None:
        return
    now = time.time()
    dt = min(max(now - gui._camera_drive_time, 0.0), 0.25)
    gui._camera_drive_time = now
    # A free, unlocked camera has nothing to say after it is activated: leaving
    # it alone keeps the mouse working
    if (
        camera.placement == PLACEMENT_FREE
        and camera.aim == AIM_FREE
    ):
        return
    apply_camera(gui, camera, dt)


# ------------------------------------------------------------------ interface


def _target_names(gui: "SionnaRtGui") -> list[str]:
    """Everything a camera can follow or lock onto."""
    names = list(gui.scene.transmitters) + list(gui.scene.receivers)
    names += [name for name in gui.scene.objects if name not in names]
    return names


def _combo(label: str, options: list[str], current: str) -> str:
    index = options.index(current) if current in options else 0
    changed, chosen = psim.Combo(label, index, options)
    return options[chosen] if changed else current


def _row(gui: "SionnaRtGui", label: str) -> None:
    property_row(label, gui.ui_scale)


def camera_contents(gui: "SionnaRtGui", camera: ViewCamera) -> None:
    """Properties of one camera."""
    is_active = gui.active_camera is camera
    if is_active:
        psim.TextColored((*ACCENT_BRIGHT, 1.0), f"{camera.name} is driving the view")
    else:
        psim.TextDisabled(f"{camera.name} is not driving the view")

    if psim.Button("Stop driving" if is_active else "Look through"):
        activate(gui, None if is_active else camera)
    psim.SameLine()
    if psim.Button("Set from current view"):
        capture_current_view(gui, camera)
    if psim.IsItemHovered():
        psim.SetTooltip("Move this camera to where you are looking now")
    # On its own row: the three together do not fit the panel
    if psim.Button("Remove##camera"):
        remove_camera(gui, camera)
        return

    psim.Separator()

    _row(gui, "Placement")
    camera.placement = _combo("##placement", PLACEMENTS, camera.placement)
    end_property_row()

    targets = _target_names(gui)
    if camera.placement == PLACEMENT_CARRIED:
        _row(gui, "Carried by")
        options = ["(nothing)"] + targets
        chosen = _combo("##carrier", options, camera.carrier or "(nothing)")
        camera.carrier = None if chosen == "(nothing)" else chosen
        end_property_row()

        _row(gui, "Offset [m]")
        _, offset = numeric_field3("##offset", camera.offset, 0.5)
        camera.offset = np.asarray(offset, dtype=np.float64)
        end_property_row()
        psim.PushTextWrapPos(0.0)
        psim.TextDisabled(
            "The camera keeps this offset from the object, so a height with no "
            "sideways part gives a fly-over."
        )
        psim.PopTextWrapPos()
    elif camera.placement == PLACEMENT_ORBIT:
        _row(gui, "Centre")
        _, centre = numeric_field3("##orbit_center", camera.orbit_center, 0.5)
        camera.orbit_center = np.asarray(centre, dtype=np.float64)
        end_property_row()

        _row(gui, "Radius [m]")
        _, camera.orbit_radius = numeric_field(
            "##orbit_radius", camera.orbit_radius, 0.5, 1.0, 1e6
        )
        end_property_row()

        _row(gui, "Height [m]")
        _, camera.orbit_height = numeric_field(
            "##orbit_height", camera.orbit_height, 0.5, -1e6, 1e6
        )
        end_property_row()

        _row(gui, "Speed [deg/s]")
        _, camera.orbit_speed_deg_s = numeric_field(
            "##orbit_speed", camera.orbit_speed_deg_s, 0.5, -360.0, 360.0
        )
        end_property_row()

        if psim.Button("Centre on scene"):
            camera.orbit_center = np.asarray(gui.scene_center(), dtype=np.float64)
        psim.SameLine()
        psim.TextDisabled(f"at {camera.orbit_angle_deg:.0f} deg")
    else:
        _row(gui, "Position")
        _, position = numeric_field3("##position", camera.position, 0.5)
        camera.position = np.asarray(position, dtype=np.float64)
        end_property_row()

        _row(gui, "Looks at")
        _, target = numeric_field3("##target", camera.target, 0.5)
        camera.target = np.asarray(target, dtype=np.float64)
        end_property_row()

    _row(gui, "Aim")
    camera.aim = _combo("##aim", AIMS, camera.aim)
    end_property_row()
    if camera.aim == AIM_LOCKED:
        _row(gui, "Locked to")
        options = ["(nothing)"] + targets
        chosen = _combo("##locked_to", options, camera.locked_to or "(nothing)")
        camera.locked_to = None if chosen == "(nothing)" else chosen
        end_property_row()
        psim.PushTextWrapPos(0.0)
        psim.TextDisabled(
            "The camera turns to keep this object in frame as it moves, the way "
            "a device locked to another one does."
        )
        psim.PopTextWrapPos()


def cameras_contents(gui: "SionnaRtGui") -> None:
    """The list of cameras, for the render tab."""
    if psim.Button("Add camera on this view"):
        camera = add_camera(gui)
        gui.select_camera(camera)
    psim.SameLine()
    if gui.active_camera is not None:
        if psim.Button("Back to mouse"):
            activate(gui, None)
        if psim.IsItemHovered():
            psim.SetTooltip("Stop driving the view from a camera")
    else:
        psim.TextDisabled("the mouse is driving the view")

    psim.Spacing()
    psim.PushTextWrapPos(0.0)
    psim.TextDisabled(
        "A camera holds a view to come back to, follows an object, or circles "
        "the scene."
    )
    psim.PopTextWrapPos()
    for camera in list(gui.cameras):
        is_active = gui.active_camera is camera
        label = f"{'> ' if is_active else ''}{camera.name}"
        if psim.Selectable(label, gui.selected_camera is camera):
            gui.select_camera(camera)
        psim.SameLine()
        psim.TextDisabled(
            camera.placement
            if camera.placement != PLACEMENT_FREE
            else ("locked" if camera.aim == AIM_LOCKED else "free")
        )

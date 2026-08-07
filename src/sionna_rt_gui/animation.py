#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field
import time
import numpy as np

import drjit as dr
import polyscope as ps
from polyscope import imgui as psim
from sionna import rt
from sionna.rt.utils.render import scene_scale

from .config import DEFAULT_SLICE_PLANE_NAME
from .sionna_utils import set_or_update_radio_devices_polyscope


class LoopingMode(Enum):
    NoLoop = 0
    Mirror = 1
    Repeat = 2


LOOPING_MODE_NAMES = ["None", "Mirror", "Repeat"]
assert len(LOOPING_MODE_NAMES) == len(LoopingMode)

SPEED_BUTTON_COLOR = np.array((0.269, 0.474, 0.377, 1.0))


@dataclass(kw_only=True)
class Trajectory:
    # Whether to enable the trajectory
    enabled: bool = False
    # Distance along the trajectory [m].
    distance: float = 0.0
    # Control points defining the polyline of the trajectory
    # TODO: consider supporting changes in orientation as well (requires fancier interpolation)
    points: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    # Movement velocity [m/s]. Default is set based on a walking speed of 4 km/h.
    velocity: float = 1.11
    # Rotate the device so it faces its direction of travel
    orient_along_path: bool = True
    # Looping mode index
    looping_mode_i: int = LoopingMode.Mirror.value
    # Whether the trajectory is currently playing in reverse, due e.g. to mirror looping mode.
    backward: bool = False

    # Cumulative distribution of distances along the trajectory, starting at zero.
    # Has width equal to the number of points.
    _cumulative_distances: list[float] = field(default_factory=list)

    @property
    def looping_mode(self) -> LoopingMode:
        return LoopingMode(self.looping_mode_i)

    def current_position_and_direction(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the current world-space position based on the distance along the trajectory."""
        n_points = len(self.points)
        if n_points == 0:
            return None
        if n_points == 1:
            return self.points[0], np.zeros(3)

        safe_dist = np.clip(self.distance, 0.0, self.total_distance())
        end_idx = np.searchsorted(self._cumulative_distances, safe_dist, side="left")
        start_idx = max(end_idx - 1, 0)
        if start_idx == end_idx:
            direction = np.zeros(3)
            return self.points[start_idx], direction

        start_dist = self._cumulative_distances[start_idx]
        end_dist = self._cumulative_distances[end_idx]
        t = (safe_dist - start_dist) / (end_dist - start_dist)

        direction = self.points[end_idx] - self.points[start_idx]
        pos = self.points[start_idx] + t * direction
        return pos, direction / np.linalg.norm(direction)

    def add_point(self, point: np.ndarray | list[float]):
        point = np.array(point)
        assert point.size == 3, "Point must be a 3D vector"

        if len(self.points) == 0:
            self.points = point.squeeze()[None, :]
            self._cumulative_distances = [0.0]
        else:
            self.points = np.concatenate(
                [self.points, point.squeeze()[None, :]], axis=0
            )
            dist_to_new = np.linalg.norm(self.points[-1] - self.points[-2])
            self._cumulative_distances.append(
                self._cumulative_distances[-1] + dist_to_new
            )

        # Snap to the latest added point
        self.distance = self._cumulative_distances[-1]

    def total_distance(self) -> float:
        if len(self._cumulative_distances) == 0:
            return 0.0
        return self._cumulative_distances[-1]

    def set_point(self, index: int, point: np.ndarray | list[float]):
        point = np.array(point).squeeze()
        assert point.size == 3, "Point must be a 3D vector"
        self.points[index] = point
        self._recompute_distances()

    def remove_point(self, index: int):
        self.points = np.delete(self.points, index, axis=0)
        self._recompute_distances()

    def _recompute_distances(self):
        n_points = len(self.points)
        if n_points == 0:
            self._cumulative_distances = []
            self.distance = 0.0
            return
        distances = [0.0]
        for i in range(1, n_points):
            distances.append(
                distances[-1]
                + float(np.linalg.norm(self.points[i] - self.points[i - 1]))
            )
        self._cumulative_distances = distances
        self.distance = float(np.clip(self.distance, 0.0, self.total_distance()))

    def clear(self):
        self.points = np.array([], dtype=float)
        self.distance = 0.0
        self.backward = False
        self._cumulative_distances.clear()

    def __len__(self) -> int:
        return len(self.points)


@dataclass(kw_only=True)
class AnimationConfig:
    playing: bool = True
    speed_multiplier: float = 1.0

    # Time at which the animation started playing the first time (Unix timestamp)
    time_started: float | None = None

    trajectories: dict[str, Trajectory] = field(
        default_factory=lambda: defaultdict(Trajectory)
    )

    def clear(self):
        self.trajectories.clear()
        self.time_started = None


def propagate_device_updates(gui: "SionnaRtGui", tx_changed: bool, rx_changed: bool):
    """
    Refresh the polyscope structures and radio results after devices moved.
    """
    if not tx_changed and not rx_changed:
        return
    if tx_changed:
        set_or_update_radio_devices_polyscope(
            gui.scene.transmitters,
            is_transmitter=True,
            gui=gui,
        )
        # Note: receivers don't affect radio maps.
        gui.reset_radio_map()
    if rx_changed:
        set_or_update_radio_devices_polyscope(
            gui.scene.receivers,
            is_transmitter=False,
            gui=gui,
        )

    if gui.cfg.paths.auto_update:
        gui.update_paths(show=True)


def restart_trajectories(gui: "SionnaRtGui"):
    """
    Move every animated device back to the start of its trajectory.
    """
    tx_changed = False
    rx_changed = False
    for obj_name, traj in gui.animation_config.trajectories.items():
        if len(traj) == 0:
            continue
        traj.distance = 0.0
        traj.backward = False
        obj = gui.scene.get(obj_name)
        if obj is None:
            continue
        if apply_trajectory_position(obj, traj):
            gui.update_attached_object(obj)
            if isinstance(obj, rt.Transmitter):
                tx_changed = True
            else:
                rx_changed = True
    propagate_device_updates(gui, tx_changed, rx_changed)


def animation_gui(gui: "SionnaRtGui"):
    """
    GUI for the main animation controls.
    """
    was_playing = gui.animation_config.playing
    toggled = psim.Button("Pause" if was_playing else "Resume")
    if toggled:
        gui.animation_config.playing = not gui.animation_config.playing
        if gui.animation_config.playing:
            gui.animation_config.time_started = time.time()

    psim.SameLine()
    if psim.Button("Restart##animation"):
        restart_trajectories(gui)
    if psim.IsItemHovered():
        psim.SetTooltip("Move every animated device back to the start of its path.")

    for speed in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        is_current = gui.animation_config.speed_multiplier == speed
        color = SPEED_BUTTON_COLOR.copy()
        if not is_current:
            color[:3] *= 0.5

        psim.PushStyleColor(psim.ImGuiCol_Button, color)
        if psim.Button(f"{speed}x"):
            gui.animation_config.speed_multiplier = speed
        psim.PopStyleColor()

        psim.SameLine()
    psim.Text(f"Speed: {gui.animation_config.speed_multiplier:.1f}x")


def apply_trajectory_position(object: rt.SceneObject, traj: Trajectory) -> bool:
    """
    Snap the object to its current position (and, if enabled, orientation)
    along the trajectory. Returns True if the object was moved.
    """
    result = traj.current_position_and_direction()
    if result is None:
        return False
    position, direction = result
    object.position = position
    object.velocity = direction * traj.velocity
    if traj.orient_along_path and np.linalg.norm(direction) > 0:
        object.look_at(position + direction)
        dr.make_opaque(object.orientation)
    dr.make_opaque(object.position, object.velocity)
    return True


def trajectory_gui(gui: "SionnaRtGui", object: rt.SceneObject) -> bool:
    """
    GUI to edit the animation trajectory & velocity of a selected object.
    Returns True if the object was moved, meaning that radio results
    (radio map, paths) should be updated.
    """
    traj = gui.animation_config.trajectories[object.name]
    moved = False

    # Reserve room for the widget labels, which would otherwise be
    # clipped when the window is narrow.
    psim.PushItemWidth(-140 * gui.ui_scale)

    if psim.Button("Add waypoint##trajectory"):
        # Prevent the trajectory from playing while we are editing,
        # otherwise the point will keep snapping back.
        traj.enabled = False
        traj.add_point(object.position.numpy())
    psim.SameLine()
    psim.TextDisabled("(at the device's position)")
    if psim.IsItemHovered():
        psim.SetTooltip(
            "Move the device (drag its gizmo or edit its position),\n"
            "then add waypoints one by one to draw a path."
        )

    n_points = len(traj)
    has_points = n_points > 0

    # Editable list of waypoints
    remove_index = None
    if has_points and psim.TreeNodeEx(
        f"Waypoints ({n_points})###trajectory_waypoints",
        psim.ImGuiTreeNodeFlags_DefaultOpen,
    ):
        for i in range(n_points):
            psim.PushID(i)
            changed, new_point = psim.DragFloat3(
                "##waypoint", tuple(traj.points[i]), 0.5, format="%.1f"
            )
            if changed:
                # Pause this trajectory while editing, otherwise the playback
                # keeps fighting the user's edits.
                traj.enabled = False
                traj.set_point(i, new_point)
                moved |= apply_trajectory_position(object, traj)
            psim.SameLine()
            if psim.SmallButton("x##remove_waypoint"):
                remove_index = i
            psim.PopID()
        psim.TreePop()

    if remove_index is not None:
        traj.remove_point(remove_index)
        n_points = len(traj)
        has_points = n_points > 0
        if has_points:
            moved |= apply_trajectory_position(object, traj)

    if has_points:
        if psim.Button("Clear all waypoints##trajectory"):
            traj.clear()
            has_points = False

    psim.BeginDisabled(not has_points)
    changed, traj.orient_along_path = psim.Checkbox(
        "Face along path##trajectory", traj.orient_along_path
    )
    if psim.IsItemHovered():
        psim.SetTooltip(
            "Rotate the device so it faces its direction of travel,\n"
            "like a vehicle. The antenna pattern rotates with it."
        )
    if changed and has_points:
        moved |= apply_trajectory_position(object, traj)

    changed, traj.enabled = psim.Checkbox("Play##trajectory", traj.enabled)
    if changed and traj.enabled:
        # Make sure the global animation is running, otherwise enabling
        # this trajectory would appear to do nothing.
        if not gui.animation_config.playing:
            gui.animation_config.playing = True
            gui.animation_config.time_started = time.time()
    psim.SameLine()
    if psim.Button("Restart##trajectory"):
        traj.distance = 0.0
        traj.backward = False
        moved |= apply_trajectory_position(object, traj)
    if traj.enabled and not gui.animation_config.playing:
        psim.SameLine()
        psim.TextDisabled("(paused, see the Animation section)")

    # Scrub along the trajectory
    changed, new_distance = psim.SliderFloat(
        "Position [m]", traj.distance, 0.0, traj.total_distance()
    )
    if changed:
        # Pause this trajectory so playback doesn't fight the scrubbing.
        traj.enabled = False
        traj.backward = False
        traj.distance = float(np.clip(new_distance, 0.0, traj.total_distance()))
        moved |= apply_trajectory_position(object, traj)

    _, traj.velocity = psim.SliderFloat("Velocity [m/s]", traj.velocity, 0.1, 30.0)

    _, traj.looping_mode_i = psim.Combo(
        "Loop##trajectory", traj.looping_mode_i, LOOPING_MODE_NAMES
    )
    if traj.looping_mode != LoopingMode.Mirror:
        # Only the mirror mode can play in reverse
        traj.backward = False
    psim.EndDisabled()

    if has_points:
        # Draw a preview of the trajectory: the path itself and its waypoints
        display_radius = max(0.0003 * scene_scale(gui.scene), 0.3)
        struct = ps.register_curve_network(
            "Trajectory",
            traj.points,
            edges="line",
            enabled=True,
            color=(0.35, 0.15, 0.25),
            transparency=0.7,
        )
        struct.set_radius(display_radius, relative=False)
        struct.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME, True)

        waypoints = ps.register_point_cloud(
            "Trajectory waypoints",
            traj.points,
            enabled=True,
            color=(0.55, 0.25, 0.40),
            transparency=0.85,
        )
        waypoints.set_radius(1.8 * display_radius, relative=False)
        waypoints.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME, True)
    else:
        if ps.has_curve_network("Trajectory"):
            ps.get_curve_network("Trajectory").remove()
        if ps.has_point_cloud("Trajectory waypoints"):
            ps.get_point_cloud("Trajectory waypoints").remove()

    if moved:
        # Carry along any scene object attached to this device
        gui.update_attached_object(object)

    psim.PopItemWidth()
    return moved


def animation_tick(gui: "SionnaRtGui", time_delta: float, force: bool = False):
    """
    Tick the animation.
    """
    cfg = gui.animation_config
    if not cfg.playing and not force:
        return

    tx_changed = False
    rx_changed = False
    for obj_name, traj in gui.animation_config.trajectories.items():
        if not traj.enabled and not force:
            continue
        if len(traj) == 0:
            continue

        distance_delta = time_delta * cfg.speed_multiplier * traj.velocity
        total_distance = traj.total_distance()
        traj.distance += (-1 if traj.backward else 1) * distance_delta

        if traj.distance <= 0:
            match traj.looping_mode:
                case LoopingMode.NoLoop:
                    traj.distance = 0.0
                case LoopingMode.Mirror:
                    traj.backward = False
                case LoopingMode.Repeat:
                    traj.distance = total_distance
                case _:
                    raise ValueError(f"Invalid looping mode: {traj.looping_mode}")
        elif traj.distance > total_distance:
            match traj.looping_mode:
                case LoopingMode.NoLoop:
                    traj.distance = total_distance
                case LoopingMode.Mirror:
                    traj.backward = True
                case LoopingMode.Repeat:
                    traj.distance = 0.0
                case _:
                    raise ValueError(f"Invalid looping mode: {traj.looping_mode}")

        # Protection in case of large single-frame jumps
        traj.distance = np.clip(traj.distance, 0.0, total_distance)

        # Update the object position (and possibly orientation) accordingly
        obj = gui.scene.get(obj_name)
        position, direction = traj.current_position_and_direction()
        obj.position = position
        # Velocity for doppler
        obj.velocity = direction * traj.velocity
        if traj.orient_along_path and np.linalg.norm(direction) > 0:
            obj.look_at(position + direction)
            dr.make_opaque(obj.orientation)
        dr.make_opaque(obj.position, obj.velocity)
        # Carry along any scene object attached to this device
        gui.update_attached_object(obj)
        if isinstance(obj, rt.Transmitter):
            tx_changed = True
        elif isinstance(obj, rt.Receiver):
            rx_changed = True

    propagate_device_updates(gui, tx_changed, rx_changed)

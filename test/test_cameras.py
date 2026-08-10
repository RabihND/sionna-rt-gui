#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Where a camera ends up, given its placement and aim."""
import numpy as np
import pytest

from sionna_rt_gui.cameras import (
    AIM_FREE,
    AIM_LOCKED,
    PLACEMENT_CARRIED,
    PLACEMENT_FREE,
    PLACEMENT_ORBIT,
    ViewCamera,
    resolved_pose,
)


class FakeGui:
    """Just enough of the app for the pose maths: where things are."""

    def __init__(self, positions=None):
        self.positions = positions or {}

    def look_at_target_position(self, name):
        return self.positions.get(name)

    def scene_center(self):
        return np.zeros(3)


def test_free_camera_keeps_its_pose():
    camera = ViewCamera(
        name="c",
        position=np.array([10.0, 20.0, 30.0]),
        target=np.array([1.0, 2.0, 3.0]),
    )
    position, target = resolved_pose(FakeGui(), camera, 0.1)
    assert np.allclose(position, [10.0, 20.0, 30.0])
    assert np.allclose(target, [1.0, 2.0, 3.0])


def test_orbit_advances_by_speed_and_holds_its_radius():
    camera = ViewCamera(
        name="c",
        placement=PLACEMENT_ORBIT,
        orbit_center=np.array([5.0, 5.0, 0.0]),
        orbit_radius=100.0,
        orbit_height=40.0,
        orbit_speed_deg_s=90.0,
        orbit_angle_deg=0.0,
    )
    gui = FakeGui()

    # A quarter turn per second at 90 deg/s
    position, target = resolved_pose(gui, camera, 1.0)
    assert camera.orbit_angle_deg == pytest.approx(90.0)
    assert np.allclose(position, [5.0, 105.0, 40.0])
    # It always looks at the centre it is circling
    assert np.allclose(target, [5.0, 5.0, 0.0])

    # And stays that far out, whatever the angle
    for _ in range(7):
        position, _ = resolved_pose(gui, camera, 0.37)
        radius = np.linalg.norm(position[:2] - camera.orbit_center[:2])
        assert radius == pytest.approx(100.0)
        assert position[2] == pytest.approx(40.0)


def test_orbit_angle_wraps():
    camera = ViewCamera(
        name="c",
        placement=PLACEMENT_ORBIT,
        orbit_speed_deg_s=300.0,
        orbit_angle_deg=200.0,
    )
    resolved_pose(FakeGui(), camera, 1.0)
    assert camera.orbit_angle_deg == pytest.approx(140.0)


def test_carried_camera_keeps_its_offset_and_watches_the_carrier():
    camera = ViewCamera(
        name="c",
        placement=PLACEMENT_CARRIED,
        carrier="rx-0",
        offset=np.array([0.0, -20.0, 35.0]),
    )
    gui = FakeGui({"rx-0": np.array([100.0, 50.0, 1.5])})
    position, target = resolved_pose(gui, camera, 0.1)
    assert np.allclose(position, [100.0, 30.0, 36.5])
    assert np.allclose(target, [100.0, 50.0, 1.5])

    # The offset is kept as the carrier moves
    gui.positions["rx-0"] = np.array([120.0, 50.0, 1.5])
    position, target = resolved_pose(gui, camera, 0.1)
    assert np.allclose(position, [120.0, 30.0, 36.5])
    assert np.allclose(target, [120.0, 50.0, 1.5])


def test_carried_camera_falls_back_to_its_own_pose_without_a_carrier():
    camera = ViewCamera(
        name="c",
        position=np.array([7.0, 7.0, 7.0]),
        placement=PLACEMENT_CARRIED,
        carrier="gone",
    )
    position, _ = resolved_pose(FakeGui(), camera, 0.1)
    assert np.allclose(position, [7.0, 7.0, 7.0])


def test_locked_aim_wins_over_the_carrier():
    camera = ViewCamera(
        name="c",
        placement=PLACEMENT_CARRIED,
        carrier="rx-0",
        offset=np.array([0.0, 0.0, 50.0]),
        aim=AIM_LOCKED,
        locked_to="tx-0",
    )
    gui = FakeGui(
        {"rx-0": np.array([10.0, 0.0, 1.5]), "tx-0": np.array([-30.0, 40.0, 12.0])}
    )
    position, target = resolved_pose(gui, camera, 0.1)
    # Carried by the receiver, but pointed at the transmitter
    assert np.allclose(position, [10.0, 0.0, 51.5])
    assert np.allclose(target, [-30.0, 40.0, 12.0])


def test_locked_aim_from_a_fixed_position_tracks_a_moving_target():
    camera = ViewCamera(
        name="c",
        position=np.array([0.0, 0.0, 30.0]),
        placement=PLACEMENT_FREE,
        aim=AIM_LOCKED,
        locked_to="rx-0",
    )
    gui = FakeGui({"rx-0": np.array([50.0, 0.0, 1.5])})
    position, target = resolved_pose(gui, camera, 0.1)
    assert np.allclose(position, [0.0, 0.0, 30.0])
    assert np.allclose(target, [50.0, 0.0, 1.5])

    gui.positions["rx-0"] = np.array([50.0, 60.0, 1.5])
    position, target = resolved_pose(gui, camera, 0.1)
    # The camera has not moved, only turned
    assert np.allclose(position, [0.0, 0.0, 30.0])
    assert np.allclose(target, [50.0, 60.0, 1.5])


def test_missing_lock_target_leaves_the_target_alone():
    camera = ViewCamera(
        name="c",
        target=np.array([3.0, 3.0, 3.0]),
        aim=AIM_LOCKED,
        locked_to="not-here",
    )
    _, target = resolved_pose(FakeGui(), camera, 0.1)
    assert np.allclose(target, [3.0, 3.0, 3.0])


def test_free_aim_on_a_free_camera_is_not_orbiting():
    camera = ViewCamera(name="c", placement=PLACEMENT_FREE, aim=AIM_FREE)
    before = camera.orbit_angle_deg
    resolved_pose(FakeGui(), camera, 5.0)
    assert camera.orbit_angle_deg == before
